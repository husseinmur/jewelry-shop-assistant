"""
LangChain-based RAG system for improved jewelry search
Provides better Arabic language understanding and semantic search
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import openai


class ArabicJewelryRAG:
    """LangChain-based RAG system for Arabic jewelry queries"""

    def __init__(self, pinecone_index, openai_api_key: str):
        """Initialize the RAG system"""
        self.pinecone_index = pinecone_index
        self.openai_api_key = openai_api_key

        # Initialize embeddings - must match existing Pinecone index dimensions (1536)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",  # 1536 dimensions - matches existing index
            openai_api_key=openai_api_key
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=openai_api_key
        )

        # Initialize vector store wrapper
        self.vector_store = None
        self.retriever = None
        self._setup_retriever()

    def _setup_retriever(self):
        """Setup the hybrid retriever (vector + BM25)"""
        try:
            # Get all documents from Pinecone
            docs = self._fetch_all_documents()

            if not docs:
                st.warning("⚠️ No documents found in vector store")
                return

            # Create Pinecone vector store wrapper
            self.vector_store = PineconeVectorStore(
                index=self.pinecone_index,
                embedding=self.embeddings,
                text_key="description"
            )

            # Create vector retriever
            vector_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            )

            # Create BM25 retriever for keyword matching
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 8

            # Combine both retrievers
            self.retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.7, 0.3]  # Favor semantic over keyword
            )

        except Exception as e:
            st.error(f"❌ Failed to setup retriever: {e}")

    def _fetch_all_documents(self) -> List[Document]:
        """Fetch all products as LangChain documents"""
        try:
            # Query Pinecone for all products
            results = self.pinecone_index.query(
                vector=[0.0] * 1536,  # Dummy vector
                top_k=1000,  # Get many results
                include_metadata=True
            )

            documents = []
            for match in results.matches:
                metadata = match.metadata

                # Create rich text content for better search
                content = self._create_document_content(metadata)

                doc = Document(
                    page_content=content,
                    metadata={
                        "id": match.id,
                        "name": metadata.get("name", ""),
                        "category": metadata.get("category", ""),
                        "price": metadata.get("price", 0),
                        "karat": metadata.get("karat", ""),
                        "weight": metadata.get("weight", 0),
                        "design": metadata.get("design", ""),
                        "style": metadata.get("style", ""),
                        "product_url": metadata.get("product_url", ""),
                        "score": getattr(match, 'score', 0)
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            st.error(f"Error fetching documents: {e}")
            return []

    def _create_document_content(self, metadata: Dict) -> str:
        """Create rich searchable content from product metadata"""
        content_parts = []

        # Product name
        if metadata.get("name"):
            content_parts.append(f"المنتج: {metadata['name']}")

        # Category
        if metadata.get("category"):
            content_parts.append(f"النوع: {metadata['category']}")

        # Materials and specifications
        if metadata.get("karat"):
            content_parts.append(f"العيار: {metadata['karat']}")

        if metadata.get("weight", 0) > 0:
            content_parts.append(f"الوزن: {metadata['weight']} جرام")

        if metadata.get("design"):
            content_parts.append(f"التصميم: {metadata['design']}")

        if metadata.get("style"):
            content_parts.append(f"الطراز: {metadata['style']}")

        # Main description
        if metadata.get("description"):
            content_parts.append(f"الوصف: {metadata['description']}")

        # Price
        if metadata.get("price"):
            content_parts.append(f"السعر: {metadata['price']} ريال")

        return " | ".join(content_parts)

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search for products using LangChain RAG"""
        try:
            if not self.retriever:
                return []

            # Enhanced query processing
            processed_query = self._enhance_query(query)

            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(processed_query)

            # Convert back to our format
            results = []
            for doc in docs[:max_results]:
                # Create result object similar to Pinecone format
                result = {
                    'id': doc.metadata.get('id', ''),
                    'score': doc.metadata.get('score', 0.5),  # Default score
                    'metadata': {
                        'name': doc.metadata.get('name', ''),
                        'category': doc.metadata.get('category', ''),
                        'price': doc.metadata.get('price', 0),
                        'karat': doc.metadata.get('karat', ''),
                        'weight': doc.metadata.get('weight', 0),
                        'design': doc.metadata.get('design', ''),
                        'style': doc.metadata.get('style', ''),
                        'product_url': doc.metadata.get('product_url', ''),
                        'description': doc.page_content
                    }
                }
                results.append(result)

            return results

        except Exception as e:
            st.error(f"Search error: {e}")
            return []

    def _enhance_query(self, query: str) -> str:
        """Enhance query for better Arabic search"""
        # Expand common Arabic jewelry terms
        expansions = {
            "سلاسل": "سلاسل عقود قلائد سلسلة قلادة عقد",
            "خواتم": "خواتم خاتم دبل",
            "اساور": "اساور سوار اسورة",
            "اقراط": "اقراط قرط حلق",
            "ذهب": "ذهب ذهبي ذهبية",
            "فضة": "فضة فضي فضية",
            "بسيط": "بسيط بساطة ناعم",
            "فاخر": "فاخر فخم راقي أنيق"
        }

        enhanced_query = query
        for term, expansion in expansions.items():
            if term in query:
                enhanced_query = f"{query} {expansion}"
                break

        return enhanced_query

    def conversational_search(self, query: str, conversation_history: List = None) -> tuple:
        """
        Perform conversational search with context awareness
        Returns: (answer, search_results)
        """
        try:
            # Get search results
            search_results = self.search(query, max_results=5)

            if not search_results:
                return "عذراً، لا توجد منتجات مطابقة لطلبك في مخزوننا الحالي. جرب مصطلحات أخرى أو تصفح مجموعتنا.", []

            # Create context from search results
            context = self._create_context_from_results(search_results)

            # Build conversation context
            conversation_context = ""
            if conversation_history:
                recent_messages = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
                for msg in recent_messages:
                    if msg.get("role") in ["user", "assistant"]:
                        content = msg.get("content", "")[:150]  # Limit length
                        conversation_context += f"{msg['role']}: {content}\n"

            # Create conversational prompt
            prompt = ChatPromptTemplate.from_template("""
أنت مساعد مبيعات ودود ومتحمس في متجر مجوهرات! 💎
تحب مساعدة العملاء في العثور على أجمل القطع التي تناسبهم.

{history_section}

قاعدة بيانات المنتجات:
{context}

شخصيتك وأسلوبك:
- مرحب وودود في جميع الأوقات
- متحمس لعرض المنتجات الجميلة
- تستخدم عبارات دافئة ومشجعة
- تهتم حقاً بإسعاد العميل
- تحافظ على استمرارية المحادثة

قواعد مهمة:
1. استخدم فقط المعلومات الموجودة في قاعدة البيانات أعلاه
2. لا تخترع أو تضيف معلومات غير موجودة
3. إذا لم تجد معلومة محددة، قل "هذه المعلومة غير متوفرة" بطريقة ودودة
4. اذكر أسماء المنتجات والأسعار عند التوصية مع إظهار الحماس
5. تأكد من إدراج الرابط إذا كان متوفراً للمنتج
6. ابدأ إجاباتك بترحيب دافئ واختتمها بعرض مساعدة إضافية
7. استخدم عبارات مثل "يسعدني أن أساعدك" و "أتمنى أن تنال إعجابك"
8. اربط إجابتك بالمحادثة السابقة عند الحاجة

سؤال العميل: {question}

إجابتك الودودة:
""")

            # Generate response
            chain = prompt | self.llm | StrOutputParser()
            # Prepare history section
            history_section = f"محادثة سابقة:\n{conversation_context}" if conversation_context else "بداية محادثة جديدة"

            response = chain.invoke({
                "context": context,
                "question": query,
                "history_section": history_section
            })

            return response, search_results

        except Exception as e:
            return f"عذراً، حدث خطأ في معالجة طلبك: {e}", []

    def _create_context_from_results(self, results: List[Dict]) -> str:
        """Create formatted context from search results"""
        context_parts = []

        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            url_info = f"- الرابط: {metadata.get('product_url')}" if metadata.get('product_url') else ""

            context_parts.append(f"""
المنتج {i}:
- الاسم: {metadata.get('name', 'غير محدد')}
- السعر: {metadata.get('price', 'غير محدد')} ريال
- النوع: {metadata.get('category', 'غير محدد')}
- العيار: {metadata.get('karat', 'غير محدد')}
- الوزن: {metadata.get('weight', 'غير محدد')} جرام
- التصميم: {metadata.get('design', 'غير محدد')}
- الطراز: {metadata.get('style', 'غير محدد')}{url_info}
- الوصف: {metadata.get('description', 'غير محدد')[:200]}...
""")

        return "\n".join(context_parts)


def init_langchain_rag(pinecone_index, openai_api_key: str) -> ArabicJewelryRAG:
    """Initialize and return LangChain RAG system"""
    try:
        rag_system = ArabicJewelryRAG(pinecone_index, openai_api_key)
        return rag_system
    except Exception as e:
        st.error(f"Failed to initialize LangChain RAG: {e}")
        return None