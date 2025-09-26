import streamlit as st
from PIL import Image
import uuid
import os
from shared.config import init_apis
from shared.database import store_product, get_all_products, delete_product

# Page config
st.set_page_config(
    page_title="Jewelry Shop - Admin",
    page_icon="💎",
    layout="wide"
)

st.title("💎 Jewelry Shop - Product Management")
st.sidebar.title("Admin Panel")

# Initialize APIs
try:
    openai_client, pinecone_index = init_apis()
    st.sidebar.success("✅ Connected to databases")
except Exception as e:
    st.sidebar.error(f"❌ Connection error: {e}")
    st.stop()

# Sidebar navigation
page = st.sidebar.selectbox("Choose action:", ["Add Products", "View Products", "Bulk Upload"])

if page == "Add Products":
    st.header("Add New Product")
    
    # Upload image
    uploaded_file = st.file_uploader(
        "Upload product image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the jewelry piece"
    )
    
    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Product Image", use_column_width=True)
        
        with col2:
            # Product details form
            with st.form("product_form"):
                name = st.text_input(
                    "Product Name*",
                    placeholder="e.g., Silver Rose Petal Earrings"
                )
                
                price = st.number_input(
                    "Price ($)*",
                    min_value=0.01,
                    step=0.01,
                    format="%.2f"
                )
                
                category = st.selectbox(
                    "Category*",
                    ["rings", "necklaces", "earrings", "bracelets", "brooches", "sets", "other"]
                )
                
                # Optional fields
                st.subheader("Optional Details")
                material = st.text_input("Material", placeholder="e.g., Sterling Silver, Gold, etc.")
                gemstones = st.text_input("Gemstones", placeholder="e.g., Diamond, Ruby, etc.")
                style = st.text_input("Style", placeholder="e.g., Vintage, Modern, Minimalist")
                
                submitted = st.form_submit_button("Add Product", type="primary")
                
                if submitted:
                    if name and price:
                        with st.spinner("Processing product..."):
                            # Save image temporarily (in production, upload to cloud storage)
                            image_filename = f"product_{uuid.uuid4()}.jpg"
                            
                            # Store product
                            success = store_product(
                                index=pinecone_index,
                                image=image,
                                name=name,
                                price=price,
                                category=category,
                                image_url=f"images/{image_filename}"  # placeholder URL
                            )
                            
                            if success:
                                st.success(f"✅ Added '{name}' to catalog!")
                                st.balloons()
                    else:
                        st.error("Please fill in required fields (Name and Price)")

elif page == "View Products":
    st.header("Product Catalog")
    
    # Fetch products
    with st.spinner("Loading products..."):
        products = get_all_products(pinecone_index, limit=50)
    
    if products:
        st.write(f"Found {len(products)} products")
        
        # Display products in grid
        cols = st.columns(3)
        
        for idx, product in enumerate(products):
            with cols[idx % 3]:
                metadata = product.metadata
                
                # Product card
                st.subheader(metadata.get('name', 'Unnamed Product'))
                st.write(f"**Price:** ${metadata.get('price', 0):.2f}")
                st.write(f"**Category:** {metadata.get('category', 'Unknown')}")
                
                # Description (truncated)
                description = metadata.get('description', '')
                if len(description) > 100:
                    description = description[:100] + "..."
                st.write(f"**Description:** {description}")
                
                # Similarity score
                st.write(f"**Match Score:** {product.score:.3f}")
                
                # Delete button
                if st.button(f"🗑️ Delete", key=f"delete_{product.id}"):
                    if delete_product(pinecone_index, product.id):
                        st.rerun()
                
                st.divider()
    else:
        st.info("No products found. Add some products first!")

elif page == "Bulk Upload":
    st.header("Bulk Upload Products")
    st.info("Upload multiple product images at once")
    
    uploaded_files = st.file_uploader(
        "Choose product images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Select multiple jewelry images to upload"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files")
        
        # Default category for bulk upload
        bulk_category = st.selectbox(
            "Default category for all items:",
            ["rings", "necklaces", "earrings", "bracelets", "brooches", "sets", "other"]
        )
        
        if st.button("Process All Images", type="primary"):
            progress_bar = st.progress(0)
            success_count = 0
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    # Process image
                    image = Image.open(uploaded_file)
                    filename_without_ext = os.path.splitext(uploaded_file.name)[0]
                    
                    # Use filename as product name (can be edited later)
                    product_name = filename_without_ext.replace('_', ' ').replace('-', ' ').title()
                    
                    # Default price (can be edited later)
                    default_price = 99.99
                    
                    # Store product
                    success = store_product(
                        index=pinecone_index,
                        image=image,
                        name=product_name,
                        price=default_price,
                        category=bulk_category,
                        image_url=f"images/{uploaded_file.name}"
                    )
                    
                    if success:
                        success_count += 1
                        
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
            
            st.success(f"✅ Successfully processed {success_count}/{len(uploaded_files)} products!")
            st.info("💡 Tip: Use 'View Products' to review and edit the auto-generated details.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Admin Tools**")
if st.sidebar.button("🔄 Refresh Database Connection"):
    st.rerun()

st.sidebar.markdown("**Quick Stats**")
try:
    all_products = get_all_products(pinecone_index, limit=1000)
    st.sidebar.metric("Total Products", len(all_products))
except:
    st.sidebar.metric("Total Products", "Error")