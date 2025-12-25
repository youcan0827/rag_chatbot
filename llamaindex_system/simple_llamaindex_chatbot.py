#!/usr/bin/env python3
"""
国交省データ RAGチャットボット - シンプルなLlamaIndex風実装
依存関係を最小限にしたシンプル版
"""

import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Hugging Face Tokenizers警告を抑制
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

load_dotenv()

class SimpleLlamaIndexChatbot:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.model = None
        self.openai_client = None
        
    def initialize(self):
        """システム初期化 - LlamaIndex風の統合API"""
        print("🦙 SimpleLlamaIndexシステムを初期化しています...")
        
        try:
            # エンベディングモデル
            print("📥 エンベディングモデルをロード中...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # OpenRouter APIクライアント
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                print("❌ OpenRouter APIキーが設定されていません")
                return False
            
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            print("✅ SimpleLlamaIndex初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ 初期化エラー: {str(e)}")
            return False
    
    def load_document(self, pdf_path):
        """文書読み込み - LlamaIndex風のDocument API"""
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
                        # LlamaIndex風のチャンクング
                        chunks = self.sentence_splitter(text, chunk_size=512, chunk_overlap=50)
                        for j, chunk in enumerate(chunks):
                            documents.append({
                                'text': chunk,
                                'metadata': {
                                    'page_label': str(i + 1),
                                    'chunk_id': j + 1,
                                    'source': os.path.basename(pdf_path)
                                }
                            })
            
            self.documents = documents
            print(f"✅ {len(documents)}個のノードを作成しました")
            return True
            
        except Exception as e:
            print(f"❌ PDF処理エラー: {str(e)}")
            return False
    
    def sentence_splitter(self, text, chunk_size=512, chunk_overlap=50):
        """LlamaIndex風のSentenceSplitter"""
        sentences = text.split('。')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence + "。") <= chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # オーバーラップ処理
                if chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + sentence + "。"
                else:
                    current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 10]
    
    def create_vector_index(self):
        """ベクトルインデックス作成 - LlamaIndex風のVectorStoreIndex"""
        print("🔄 VectorStoreIndexを作成中...")
        
        try:
            texts = [doc['text'] for doc in self.documents]
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # FAISSインデックス作成
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            # 正規化
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype('float32'))
            
            print(f"✅ VectorStoreIndex作成完了（次元: {dimension}）")
            return True
            
        except Exception as e:
            print(f"❌ インデックス作成エラー: {str(e)}")
            return False
    
    def as_query_engine(self, similarity_top_k=3):
        """クエリエンジン作成 - LlamaIndex風のQueryEngine API"""
        self.similarity_top_k = similarity_top_k
        return self
    
    def query(self, question):
        """クエリ実行 - LlamaIndex風のQuery API"""
        if not self.index:
            return "❌ インデックスが作成されていません"
        
        try:
            # 類似文書検索
            query_embedding = self.model.encode([question])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding.astype('float32'), self.similarity_top_k)
            
            # ソースノード作成
            source_nodes = []
            context_texts = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    source_nodes.append({
                        'node': {
                            'text': doc['text'],
                            'metadata': doc['metadata']
                        },
                        'score': float(score)
                    })
                    context_texts.append(doc['text'])
            
            # Tree Summarize風の回答生成
            response = self.tree_summarize(question, context_texts)
            
            # LlamaIndex風のResponse オブジェクト
            return LlamaIndexResponse(response, source_nodes)
            
        except Exception as e:
            return f"❌ クエリ実行エラー: {str(e)}"
    
    def tree_summarize(self, question, context_texts):
        """Tree Summarize風の回答生成"""
        try:
            # コンテキスト統合
            combined_context = "\n\n".join([
                f"文書{i+1}: {text}" for i, text in enumerate(context_texts)
            ])
            
            # 高度なプロンプト（LlamaIndex風）
            prompt = f"""あなたは専門的な文書分析アシスタントです。以下の文書情報を総合的に分析して、質問に詳しく日本語で回答してください。

複数の文書情報:
{combined_context}

質問: {question}

回答の指針:
1. 複数の文書から関連する情報を統合して回答
2. 文書にない情報は推測せず、「提供された文書には記載されていません」と明記
3. 可能な限り具体的で詳細な回答を提供
4. 情報源が複数ある場合は、それぞれの観点を整理して説明

回答:"""
            
            response = self.openai_client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは国土交通省の専門文書を分析する高度なアシスタントです。複数の文書を統合して包括的な回答を生成します。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"❌ 回答生成エラー: {str(e)}"
    
    def chat(self):
        """チャットループ"""
        print("\n" + "="*60)
        print("🦙 国交省データ RAGチャットボット (SimpleLlamaIndex版)")
        print("="*60)
        print("質問を入力してください。終了するには 'quit' または 'exit' を入力。")
        print("詳細情報を表示するには 'verbose' モードをオンにできます。")
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
                
                print("\n🔍 SimpleLlamaIndexで検索・回答生成中...")
                
                response = self.query(user_input)
                
                if isinstance(response, LlamaIndexResponse):
                    print(f"\n🦙 回答: {response.response}")
                    
                    if verbose and response.source_nodes:
                        print("\n📚 参照したソースノード:")
                        print("-" * 40)
                        for i, source in enumerate(response.source_nodes, 1):
                            metadata = source['node']['metadata']
                            print(f"{i}. {metadata['source']} - ページ{metadata['page_label']}-{metadata['chunk_id']} (スコア: {source['score']:.3f})")
                            print(f"   {source['node']['text'][:150]}...")
                            print()
                else:
                    print(f"\n🦙 回答: {response}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 チャットボットを終了します。")
                break
            except Exception as e:
                print(f"❌ エラー: {str(e)}")

class LlamaIndexResponse:
    """LlamaIndex風のResponseオブジェクト"""
    def __init__(self, response_text, source_nodes):
        self.response = response_text
        self.source_nodes = source_nodes
    
    def __str__(self):
        return self.response

def main():
    print("🦙 国交省データ RAGチャットボット - SimpleLlamaIndex版")
    print("=" * 55)
    
    # OpenRouter APIキー確認
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OpenRouter APIキーが設定されていません。")
        print("💡 .envファイルに OPENROUTER_API_KEY=your_api_key を設定してください。")
        return
    
    # チャットボット初期化
    chatbot = SimpleLlamaIndexChatbot()
    
    if not chatbot.initialize():
        return
    
    # PDF読み込み
    pdf_path = "/Users/yoshinomukanou/Downloads/国交省に関するデータ/国交省①.pdf"
    if not chatbot.load_document(pdf_path):
        return
    
    # ベクトルインデックス作成
    if not chatbot.create_vector_index():
        return
    
    # クエリエンジン設定
    query_engine = chatbot.as_query_engine(similarity_top_k=3)
    
    # チャット開始
    query_engine.chat()

if __name__ == "__main__":
    main()