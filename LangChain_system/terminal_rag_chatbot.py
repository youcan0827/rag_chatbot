#!/usr/bin/env python3
"""
国交省データ RAGチャットボット - 100% LangChain実装
LangChainライブラリを完全に活用したRAGシステム
"""

import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import RetrievalQA
import tempfile
import logging

# ログレベル設定
logging.basicConfig(level=logging.WARNING)

# Hugging Face Tokenizers警告を抑制
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

load_dotenv()

class LangChainRAGChatbot:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.retriever = None
        
    def initialize(self):
        """LangChainシステム初期化"""
        print("🔗 LangChainシステムを初期化しています...")
        
        try:
            # OpenAI API Key確認
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            
            if not openai_api_key and not openrouter_api_key:
                print("❌ OpenAI または OpenRouter APIキーが設定されていません")
                return False
            
            # LangChain LLM設定
            if openai_api_key:
                print("✅ OpenAI APIを使用します")
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    api_key=openai_api_key
                )
            else:
                print("✅ OpenRouter APIを使用します")
                self.llm = ChatOpenAI(
                    model="openai/gpt-3.5-turbo",
                    temperature=0.7,
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            
            # エンベディング設定（多言語対応）
            print("📥 エンベディングモデルをロード中...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            print("✅ LangChain初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ 初期化エラー: {str(e)}")
            return False
    
    def load_and_process_pdf(self, pdf_path):
        """LangChainでPDFを読み込んで処理"""
        print(f"📄 LangChainでPDFを読み込み中: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print("❌ PDFファイルが見つかりません")
            return False
        
        try:
            # LangChain PDFローダー
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"✅ {len(documents)}ページを読み込みました")
            
            # LangChain テキスト分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\\n\\n", "\\n", "。", "、", " ", ""]
            )
            
            # 文書分割
            splits = text_splitter.split_documents(documents)
            print(f"✅ {len(splits)}個のチャンクに分割しました")
            
            # サンプルチャンク表示
            if splits:
                print(f"📝 サンプルチャンク: {splits[0].page_content[:100]}...")
            
            # LangChain ベクトルストア作成（Chroma）
            print("🔄 Chromaベクトルストアを作成中...")
            
            # 一時ディレクトリでChromaDB作成
            with tempfile.TemporaryDirectory() as temp_dir:
                self.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory=temp_dir
                )
                
                # メモリ内で使用するために新しいインスタンス作成
                self.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings
                )
            
            print(f"✅ ベクトルストア作成完了（{len(splits)}ドキュメント）")
            
            # LangChain リトリーバー作成
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # デバッグ: 検索テスト
            print("🔍 ベクトルストア検索テスト...")
            test_results = self.retriever.get_relevant_documents("国土交通省")
            print(f"📊 テスト検索結果: {len(test_results)}件")
            
            return True
            
        except Exception as e:
            print(f"❌ PDF処理エラー: {str(e)}")
            return False
    
    def create_qa_chain(self):
        """LangChain QAチェーン作成"""
        print("⛓️ LangChain QAチェーンを作成中...")
        
        try:
            # カスタムプロンプトテンプレート
            template = """あなたは国土交通省の専門文書を分析するアシスタントです。
以下の文書情報に基づいて、質問に日本語で詳しく回答してください。
文書に関連する情報がない場合は、「提供された文書にはその情報が含まれておりません」と回答してください。

文書情報:
{context}

質問: {question}

回答:"""

            prompt = ChatPromptTemplate.from_template(template)
            
            # LangChain LCEL (LangChain Expression Language) チェーン
            def format_docs(docs):
                return "\\n\\n".join([
                    f"【ページ{doc.metadata.get('page', 'N/A')}】\\n{doc.page_content}" 
                    for doc in docs
                ])
            
            self.qa_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            print("✅ LangChain QAチェーン作成完了")
            return True
            
        except Exception as e:
            print(f"❌ QAチェーン作成エラー: {str(e)}")
            return False
    
    def query(self, question):
        """LangChainでクエリ実行"""
        if not self.qa_chain:
            return "❌ QAチェーンが初期化されていません"
        
        try:
            print("🔍 LangChainで検索・回答生成中...")
            
            # 関連文書取得（デバッグ用）
            relevant_docs = self.retriever.get_relevant_documents(question)
            print(f"📊 検索結果: {len(relevant_docs)}件の関連文書")
            
            for i, doc in enumerate(relevant_docs, 1):
                page = doc.metadata.get('page', 'N/A')
                print(f"   {i}. ページ{page}: {doc.page_content[:100]}...")
            
            # LangChainチェーン実行
            response = self.qa_chain.invoke(question)
            return response
            
        except Exception as e:
            return f"❌ クエリ実行エラー: {str(e)}"
    
    def query_with_source(self, question):
        """ソース情報付きでクエリ実行"""
        if not self.retriever:
            return "❌ リトリーバーが初期化されていません", []
        
        try:
            # 関連文書取得
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            # 回答生成
            answer = self.qa_chain.invoke(question) if self.qa_chain else "QAチェーンエラー"
            
            # ソース情報整理
            sources = []
            for doc in relevant_docs:
                sources.append({
                    'page': doc.metadata.get('page', 'N/A'),
                    'source': doc.metadata.get('source', 'N/A'),
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            return answer, sources
            
        except Exception as e:
            return f"❌ クエリ実行エラー: {str(e)}", []
    
    def chat(self):
        """チャットループ"""
        print("\\n" + "="*60)
        print("🔗 国交省データ RAGチャットボット (100% LangChain)")
        print("="*60)
        print("質問を入力してください。終了するには 'quit' または 'exit' を入力。")
        print("詳細情報を表示するには 'verbose' モードをオンにできます。")
        print("-"*60)
        
        verbose = False
        
        while True:
            try:
                user_input = input("\\n💬 質問: ").strip()
                
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
                
                if verbose:
                    answer, sources = self.query_with_source(user_input)
                    print(f"\\n🔗 回答: {answer}")
                    
                    if sources:
                        print("\\n📚 参照したソース:")
                        print("-" * 40)
                        for i, source in enumerate(sources, 1):
                            print(f"{i}. ページ{source['page']} - {source['source']}")
                            print(f"   {source['content']}")
                            print()
                else:
                    answer = self.query(user_input)
                    print(f"\\n🔗 回答: {answer}")
                    
            except KeyboardInterrupt:
                print("\\n\\n👋 チャットボットを終了します。")
                break
            except Exception as e:
                print(f"❌ エラー: {str(e)}")

def main():
    print("🔗 国交省データ RAGチャットボット - 100% LangChain実装")
    print("=" * 55)
    
    # APIキー確認
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OpenAI または OpenRouter APIキーが設定されていません。")
        print("💡 .envファイルに OPENAI_API_KEY または OPENROUTER_API_KEY を設定してください。")
        return
    
    # チャットボット初期化
    chatbot = LangChainRAGChatbot()
    
    if not chatbot.initialize():
        return
    
    # PDF読み込み・処理
    pdf_path = "/Users/yoshinomukanou/Downloads/国交省に関するデータ/国交省①.pdf"
    if not chatbot.load_and_process_pdf(pdf_path):
        return
    
    # QAチェーン作成
    if not chatbot.create_qa_chain():
        return
    
    # チャット開始
    chatbot.chat()

if __name__ == "__main__":
    main()