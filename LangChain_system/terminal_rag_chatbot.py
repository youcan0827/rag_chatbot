#!/usr/bin/env python3
"""
国交省データ RAGチャットボット - LangChain風実装
シンプルなターミナル版RAGチャットボット
"""

import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import sys

# Hugging Face Tokenizers警告を抑制
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

load_dotenv()

class LangChainStyleRAGChatbot:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.model = None
        self.openai_client = None
        
    def initialize(self):
        """システム初期化"""
        print("🚀 システムを初期化しています...")
        
        # エンベディングモデル
        print("📥 エンベディングモデルをロード中...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # OpenRouterクライアント
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ OpenRouter APIキーが設定されていません")
            return False
        
        self.openai_client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        print("✅ モデル初期化完了")
        return True
    
    def load_pdf(self, pdf_path):
        """PDFを読み込んでテキストを抽出"""
        print(f"📄 PDFを読み込み中: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print("❌ PDFファイルが見つかりません")
            return False
        
        try:
            documents = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        chunks = self.split_text(text, 500)
                        for j, chunk in enumerate(chunks):
                            documents.append({
                                'text': chunk,
                                'page': i + 1,
                                'chunk': j + 1,
                                'source': os.path.basename(pdf_path)
                            })
            
            self.documents = documents
            print(f"✅ {len(documents)}個のテキストチャンクを抽出しました")
            return True
        except Exception as e:
            print(f"❌ PDF読み込みエラー: {str(e)}")
            return False
    
    def split_text(self, text, max_length=500):
        """テキストを指定した長さのチャンクに分割"""
        sentences = text.split('。')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence + "。") <= max_length:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 10]
    
    def create_embeddings(self):
        """エンベディングを作成してFAISSインデックスを構築"""
        print("🔄 エンベディングを作成中...")
        
        try:
            texts = [doc['text'] for doc in self.documents]
            print(f"📊 エンベディング対象: {len(texts)}個のテキストチャンク")
            
            # サンプルテキスト表示
            if texts:
                print(f"📝 サンプルテキスト: {texts[0][:100]}...")
            
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            print(f"📏 エンベディング形状: {self.embeddings.shape}")
            
            # FAISSインデックス作成
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            # 正規化
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype('float32'))
            
            # インデックス確認
            print(f"📊 FAISSインデックス内文書数: {self.index.ntotal}")
            
            print(f"✅ エンベディング作成完了（次元: {dimension}）")
            return True
        except Exception as e:
            print(f"❌ エンベディング作成エラー: {str(e)}")
            return False
    
    def search_similar_documents(self, query, k=3):
        """クエリに類似した文書を検索"""
        try:
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            print(f"🔍 デバッグ: 検索結果 (クエリ: '{query}')")
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['score'] = float(score)
                    print(f"   {i+1}. スコア: {score:.4f}, ページ{doc['page']}-{doc['chunk']}")
                    print(f"      テキスト: {doc['text'][:100]}...")
                    results.append(doc)
            
            return results
        except Exception as e:
            print(f"❌ 検索エラー: {str(e)}")
            return []
    
    def generate_answer(self, query, context_docs):
        """OpenAI GPTを使用して回答を生成"""
        try:
            context = "\n\n".join([
                f"【ページ{doc['page']}-{doc['chunk']}】\n{doc['text']}" 
                for doc in context_docs
            ])
            
            prompt = f"""以下の文書情報に基づいて、質問に日本語で回答してください。
文書に関連する情報がない場合は、「提供された文書にはその情報が含まれておりません」と回答してください。

文書情報:
{context}

質問: {query}

回答:"""
            
            response = self.openai_client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは国土交通省の文書に基づいて質問に回答する専門アシスタントです。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ 回答生成エラー: {str(e)}"
    
    def chat(self):
        """チャットループ"""
        print("\n" + "="*60)
        print("🔗 国交省データ RAGチャットボット (LangChain風)")
        print("="*60)
        print("質問を入力してください。終了するには 'quit' または 'exit' を入力。")
        print("参照文書を表示するには 'verbose' モードをオンにできます。")
        print("-"*60)
        
        verbose = False
        
        while True:
            try:
                user_input = input("\n💬 質問: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 チャットボットを終了します。")
                    break
                
                if user_input.lower() == 'verbose':
                    verbose = not verbose
                    status = "オン" if verbose else "オフ"
                    print(f"📚 詳細表示: {status}")
                    continue
                
                if not user_input:
                    continue
                
                print("\n🔍 検索中...")
                relevant_docs = self.search_similar_documents(user_input, k=3)
                
                # 類似度が低すぎる場合のフィルタリング（閾値: 0.3）
                min_similarity = 0.3
                filtered_docs = [doc for doc in relevant_docs if doc['score'] >= min_similarity]
                
                print(f"📊 検索結果: {len(relevant_docs)}件中{len(filtered_docs)}件が閾値({min_similarity})以上")
                
                if filtered_docs:
                    print("💭 回答生成中...")
                    answer = self.generate_answer(user_input, filtered_docs)
                    
                    print(f"\n🤖 回答: {answer}")
                    
                    if verbose:
                        print("\n📚 参照した文書:")
                        print("-" * 40)
                        for i, doc in enumerate(filtered_docs, 1):
                            print(f"{i}. {doc['source']} - ページ{doc['page']}-{doc['chunk']} (類似度: {doc['score']:.3f})")
                            print(f"   {doc['text'][:150]}...")
                            print()
                else:
                    print("❌ 関連する文書が見つかりませんでした。")
                    print("💡 検索された文書の類似度が低すぎます。より具体的な質問を試してください。")
                    
            except KeyboardInterrupt:
                print("\n\n👋 チャットボットを終了します。")
                break
            except Exception as e:
                print(f"❌ エラー: {str(e)}")

def main():
    print("🔗 国交省データ RAGチャットボット - LangChain風実装")
    print("=" * 50)
    
    # OpenRouter APIキー確認
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OpenRouter APIキーが設定されていません。")
        print("💡 .envファイルに OPENROUTER_API_KEY=your_api_key を設定してください。")
        return
    
    # チャットボット初期化
    chatbot = LangChainStyleRAGChatbot()
    
    if not chatbot.initialize():
        return
    
    # PDF読み込み
    pdf_path = "/Users/yoshinomukanou/Downloads/国交省に関するデータ/国交省①.pdf"
    if not chatbot.load_pdf(pdf_path):
        return
    
    # エンベディング作成
    if not chatbot.create_embeddings():
        return
    
    # チャット開始
    chatbot.chat()

if __name__ == "__main__":
    main()