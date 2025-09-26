# 💎 AI-Powered Jewelry Shop Assistant

An intelligent Arabic chatbot system for jewelry stores with advanced search capabilities, built with Streamlit, OpenAI, and Pinecone.

## 🌟 Features

### 🤖 Multiple AI Implementations
- **LangChain RAG**: Advanced semantic search with conversational context
- **Function Calling**: Intelligent tool-based AI interactions
- **Hybrid Search**: Vector similarity + keyword matching (BM25)

### 🔍 Dual Search Capabilities
- **Text Search**: Natural Arabic language queries
- **Image Search**: Upload jewelry photos to find similar items
- **Smart Product Discovery**: AI-powered product recommendations

### 🛠️ Admin Interface
- **Product Management**: Add, view, and delete jewelry items
- **Bulk Upload**: Process multiple product images at once
- **Rich Metadata**: Track price, category, karat, weight, design, style, and URLs
- **AI-Generated Descriptions**: Automatic product descriptions from images

### 🌐 Arabic Language Support
- **Native RTL Interface**: Right-to-left layout optimized for Arabic
- **Arabic Query Processing**: Natural language understanding in Arabic
- **Localized UI**: Complete Arabic interface and responses

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key
- Pinecone API Key
- Conda (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/jewelry-shop-assistant.git
   cd jewelry-shop-assistant
   ```

2. **Create and activate environment**
   ```bash
   conda create -n jewelry-shop python=3.9
   conda activate jewelry-shop
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**

   Create `.streamlit/secrets.toml`:
   ```toml
   [secrets]
   OPENAI_API_KEY = "your-openai-api-key"
   PINECONE_API_KEY = "your-pinecone-api-key"
   PINECONE_INDEX_NAME = "jewelry-products"
   ```

### Running the Application

#### Main Chatbot (Recommended)
```bash
streamlit run chatbot_langchain_arabic.py
```

#### Admin Interface
```bash
streamlit run admin_app_arabic.py
```

#### Alternative Chatbot (Function Calling)
```bash
streamlit run chatbot_tools_arabic.py
```

## 📁 Project Structure

```
jewelry-shop-assistant/
├── 🤖 Chatbot Interfaces
│   ├── chatbot_langchain_arabic.py    # LangChain RAG implementation
│   ├── chatbot_tools_arabic.py        # Function calling approach
│   └── admin_app_arabic.py            # Product management interface
├── 📚 Shared Modules
│   ├── shared/
│   │   ├── config.py                  # API configuration
│   │   ├── database.py                # Pinecone operations
│   │   ├── embeddings.py              # AI image/text processing
│   │   └── langchain_rag.py           # LangChain RAG system
├── 🧪 Testing & Development
│   ├── test_*.py                      # Various test files
│   └── CLAUDE.md                      # Development session log
└── 📋 Configuration
    ├── requirements.txt               # Python dependencies
    ├── .gitignore                     # Git ignore rules
    └── README.md                      # This file
```

## 🔧 Key Components

### Search Architecture
```
User Query → Embedding → Vector Search → Pinecone Database
                     ↓
AI Response ← LLM Processing ← Formatted Results
```

### Three-Tier Search Strategy
1. **Primary Search** (≥50% similarity) - High quality matches
2. **Fallback Search** (≥35% similarity) - Moderate matches if primary fails
3. **Final Filter** (≥40% similarity) - Quality gate before display

## 💻 Usage Examples

### Text Search Queries (Arabic)
- "عندكن سلاسل ذهبية؟" (Do you have gold necklaces?)
- "أريد شيء أنيق للزفاف" (I want something elegant for wedding)
- "قطعة بسيطة للاستخدام اليومي" (Simple piece for daily use)

### Admin Operations
1. **Add Product**: Upload image, set price, category, and details
2. **Bulk Upload**: Process multiple images with default settings
3. **View Catalog**: Browse all products with search and filter
4. **Product URLs**: Add links to external product pages

## 🛡️ Security & Best Practices

- **API Key Protection**: Store credentials in Streamlit secrets
- **Input Validation**: Proper handling of user uploads and inputs
- **Error Handling**: Comprehensive error management and logging
- **Performance Optimization**: Efficient vector operations and caching

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Add secrets in dashboard
4. Deploy automatically

### Local Production
```bash
streamlit run chatbot_langchain_arabic.py --server.port 8501
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-4 and embedding models
- **Pinecone** for vector database services
- **LangChain** for RAG framework
- **Streamlit** for the web interface
- **Anthropic Claude** for development assistance

## 📞 Support

For questions and support:
- Create an issue in this repository
- Check the [CLAUDE.md](CLAUDE.md) file for detailed development notes

---

Built with ❤️ for jewelry businesses worldwide 💎