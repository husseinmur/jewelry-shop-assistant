import openai
import base64
import io
from PIL import Image
import streamlit as st
from .config import EMBEDDING_MODEL, VISION_MODEL, TEXT_MODEL, MAX_IMAGE_SIZE

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    """Resize image while maintaining aspect ratio and handle format conversion"""
    # Convert to RGB if necessary
    if image.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    # Convert RGBA to RGB if necessary
    if image.mode in ('RGBA', 'LA', 'P'):
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def get_image_description(image):
    """Get detailed description of jewelry image using GPT-4V"""
    try:
        # Resize and convert to base64
        resized_image = resize_image(image.copy())
        base64_image = image_to_base64(resized_image)
        
        response = openai.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """صِف هذه القطعة من المجوهرات بالتفصيل لكتالوج متجر. ركز على:
                            - نوع المجوهرات (خاتم، عقد، أقراط، إلخ)
                            - المواد (ذهب، فضة، أحجار كريمة، إلخ)  
                            - طراز التصميم (عصري، كلاسيكي، بسيط، إلخ)
                            - الميزات البارزة (أنماط، ملمس، عناصر خاصة)
                            - الجمالية العامة والجاذبية
                            
                            كن محدداً ووصفياً لمساعدة العملاء في العثور على قطع مشابهة.
                            أجب باللغة العربية."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            # max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"خطأ في الحصول على وصف الصورة: {e}")
        return "قطعة مجوهرات جميلة"

def get_image_category(image):
    """Detect jewelry category from image using GPT-4V"""
    try:
        # Resize and convert to base64
        resized_image = resize_image(image.copy())
        base64_image = image_to_base64(resized_image)
        
        response = openai.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """حدد نوع هذه المجوهرات بكلمة واحدة فقط:
                            
                            خيارات:
                            - خواتم
                            - عقود  
                            - أقراط
                            - أساور
                            - دبابيس
                            - طقم
                            - أخرى
                            
                            أجب بكلمة واحدة فقط من القائمة أعلاه."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=10
        )
        
        category = response.choices[0].message.content.strip()
        return category if category in ["خواتم", "عقود", "أقراط", "أساور", "دبابيس", "طقم", "أخرى"] else "أخرى"
        
    except Exception as e:
        st.error(f"خطأ في تحديد فئة الصورة: {e}")
        return "أخرى"

def expand_search_query(query):
    """Use GPT-4 to expand search query with related terms"""
    try:
        response = openai.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """أنت خبير في البحث عن المجوهرات. لأي استعلام بحث عن المجوهرات، ركز على عناصر التصميم والأشكال بدلاً من المطابقات الحرفية فقط:

                    لاستعلامات الشكل/التصميم (مثل "بشكل الياسمين"، "بشكل الوردة"، "هندسي"):
                    - ركز على عناصر التصميم المرئي والأنماط والأشكال
                    - اشمل مفاهيم التصميم ذات الصلة وفئات الطراز الأوسع
                    - فكر في ما يجعل هذا الشكل مميزاً
                    
                    لاستعلامات المواد/الطراز: ركز على المواد والطرازات ذات الصلة
                    
                    للفئة: حدد نوع المجوهرات المذكور (خاتم، قلادة، أقراط، إلخ). إذا لم يكن واضحاً، استخدم "مجوهرات"
                    
                    اكتب إجابتك بالتنسيق التالي:
                    الأساسي: [المفهوم الرئيسي مع التركيز على التصميم/الشكل]
                    ذات صلة: [عنصر تصميم 1]، [نمط أوسع]، [فئة الطراز]، [شكل مشابه]، [جمالية مماثلة]
                    الفئة: [نوع المجوهرات المحدد مثل: خاتم، قلادة، أقراط، أساور، دبابيس، أو "مجوهرات" إذا غير محدد]
                    
                    مثال: "قلادة بشكل الياسمين" → الفئة: قلادة
                    مثال: "مجوهرات بشكل الياسمين" → الفئة: مجوهرات
                    
                    أجب باللغة العربية فقط."""
                },
                {
                    "role": "user", 
                    "content": f"وسّع هذا البحث عن المجوهرات: '{query}'"
                }
            ],
            # max_tokens=150,
            temperature=1.0
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"خطأ في توسيع الاستعلام: {e}")
        return f"الأساسي: {query}\nذات صلة: {query}\nالفئة: مجوهرات"

def get_text_embedding(text):
    """Get OpenAI embedding for text"""
    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
        
    except Exception as e:
        st.error(f"خطأ في الحصول على تضمين النص: {e}")
        return None

def parse_query_expansion(expansion_text):
    """Parse the GPT-4 query expansion response"""
    try:
        lines = expansion_text.strip().split('\n')
        
        primary = ""
        related = []
        category = ""
        
        for line in lines:
            if line.startswith("الأساسي:"):
                primary = line.replace("الأساسي:", "").strip()
            elif line.startswith("ذات صلة:"):
                related_str = line.replace("ذات صلة:", "").strip()
                related = [term.strip() for term in related_str.split('،')]
            elif line.startswith("الفئة:"):
                category = line.replace("الفئة:", "").strip()
        
        return {
            "primary": primary,
            "related": related,
            "category": category
        }
        
    except Exception as e:
        st.error(f"خطأ في تحليل التوسع: {e}")
        return {
            "primary": "مجوهرات",
            "related": ["مجوهرات"],
            "category": "إكسسوارات"
        }