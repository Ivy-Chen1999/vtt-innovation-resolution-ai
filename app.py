#streamlit run app.py 前端全在这个app
#chatbot，项目介绍
#引导用户点击

import os
import sys
import streamlit as st
from streamlit.components.v1 import html
import warnings

# 抑制所有警告
warnings.filterwarnings("ignore")

# 尝试生成配置文件（如果文件存在）
try:
    config_generator = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_config_from_toml.py")
    if os.path.exists(config_generator):
        print("尝试生成配置文件...")
        import subprocess
        result = subprocess.run([sys.executable, config_generator], capture_output=True, text=True)
        if result.returncode == 0:
            print("配置文件生成成功！")
            print(result.stdout)
        else:
            print(f"配置文件生成失败: {result.stderr}")
    else:
        print(f"配置生成脚本不存在: {config_generator}")
except Exception as e:
    print(f"尝试生成配置文件时出错: {str(e)}")

# 创建数据目录和密钥目录（如果不存在）
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
keys_dir = os.path.join(data_dir, "keys")
os.makedirs(keys_dir, exist_ok=True)



from innovation_resolution import chat_bot

# ----------------------------
# Page Config & Title
# ----------------------------
st.set_page_config(page_title="VTT Innovation Platform", layout="wide")
st.title("🔍 VTT Innovation Knowledge Graph Platform")

# ----------------------------
# Introduction
# ----------------------------
st.markdown("""
#### Welcome to the **VTT Innovation Knowledge Graph Platform**!

#### This tool helps you explore relationships between innovations and organizations based on publicly available data
#### The platform includes:
- 🌐 **Interactive resultsizations** of innovation networks in both 2D and 3D.
- 🌟 **Statistical dashboards** that summarize key patterns and contributors.
- 🧠 A **semantic assistant** that helps you navigate the graph via natural language queries.
- 🧪 A Clustering Method Explorer for different clustering method metrics

#### Scroll down to explore each module below:
""")

from PIL import Image
import os

@st.cache_data
def load_image(path):
    return Image.open(path)

#wanchengle
# --- Display HTML with Expand Button ---
st.header("🌐 Network Graph Visualizations")


#在ein的本地
html_path = "results/innovation_network_3d.html"
if os.path.exists(html_path):
    st.subheader("Interactive Network (Before Dedupulication)")
    with open(html_path, "r", encoding="utf-8") as f:
        html(f.read(), height=600)
else:
    st.warning("3D HTML file not found. Please run the backend script to generate it.")

# --- Display HTML with Expand Button ---

st.markdown("""
These visualizations represent the relationships between innovations and organizations:

- **Blue nodes**: Innovations
- **Green nodes**: Organizations
- **Red edges**: "Developed By" relationships
- **Blue edges**: Collaborations

Hover or zoom to explore the connections. The layout is generated based on semantic clustering.
""")

html_path = "results/innovation_network_tufte_3D.html"
if os.path.exists(html_path):
    st.subheader("3D Interactive Network (After Dedupulication)")
    with open(html_path, "r", encoding="utf-8") as f:
        html(f.read(), height=600)
else:
    st.warning("3D HTML file not found. Please run the backend script to generate it.")
st.divider()



# ----------------------------
# Innovation Metrics Dashboard
# ----------------------------
st.header(" 🌟 Innovation Metrics Dashboard")
st.markdown("""
These charts summarize statistical patterns in the innovation network:

- Count of innovations
- Proportion of multi-source or multi-developer innovations
- Top contributing organizations
            
please refresh the page if a server connection error occurs, as the cache server is not working properly with Streamlit.
""")

img_path = "results/innovation_network_tufte_2D.png"
if os.path.exists(img_path):
    img = load_image(img_path)
    st.subheader("2D Network Snapshot")
    st.image(img, use_container_width=True)
else:
    st.warning("2D PNG image not found.")





# 第二行：两列展示 Statistics 和 Top Organizations
col1, col2 = st.columns(2)

with col2:
    st.subheader("Key Innovation Statistics")
    img_stat = "results/innovation_stats_tufte.png"
    if os.path.exists(img_stat):
        img = load_image(img_stat)
        st.image(img, use_container_width=True)
        st.markdown("""
        Summary statistics highlighting:
        - Total innovations in the dataset
        - Innovations sourced from multiple data providers
        - Innovations developed by more than one organization
        """)
    else:
        st.warning("Innovation stats image not found.")

with col1:
    st.subheader("Top Contributing Organizations")
    img_top_orgs = "results/top_organizations.png"
    if os.path.exists(img_top_orgs):
        img = load_image(img_top_orgs)
        st.image(img, use_container_width=True)
        st.markdown("""
        - Organizations ranked by the number of innovations they have contributed to.
        - A great way to identify major innovation players in the ecosystem.
        """)
    else:
        st.warning("Top organizations image not found.")
# ----------------------------
# Semantic Graph Assistant
# ----------------------------

#chatbot部分
# st.header("🧠 Semantic Graph Assistant")

# query = st.text_input("💬 free-form questions like 'Who developed nuclear energy innovations?', 'Which organizations developed the most innovations?':")

# if query:
#     with st.spinner("Retrieving relevant information..."):
#         reply = chat_bot(query)
#     st.success("🧠 Answer:")
#     st.markdown(reply)
#     st.info("🔎 This answer is based on the top 3 semantically similar innovation descriptions retrieved from the knowledge graph.")

# st.divider()


with st.sidebar:
    st.header("💬 Ask the AI Assistant")
    st.markdown("""
        💬 free-form questions like : Who developed nuclear energy innovations?, Which organizations developed the most innovations?
        """)
    user_input = st.chat_input("Ask something...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_bot(user_input)
                st.markdown(response)


import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Clustering Method Explorer
# ------------------------------
st.divider()
st.header("🧪 Clustering Method Explorer")

cluster_data = {
    "Threshold-based (0.85)": {"clusters": 1911, "edges": 12502, "note": "Baseline"},
    "HDBSCAN": {"clusters": 1735, "edges": 12341, "note": "Aggressive deduplication"},
    "KMeans (n=1911)": {"clusters": 1911, "edges": 12544, "note": "Balanced"},
    "Agglomerative (n=1911)": {"clusters": 1911, "edges": 12544, "note": "Similar to KMeans"},
    "Spectral (n=1911, k=15)": {"clusters": 1911, "edges": 12612, "note": "Highest edge count (dense)"}
}


method = st.selectbox("🔘 Select a clustering method", list(cluster_data.keys()), index=0)
selected = cluster_data[method]

# 卡片样式展示
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("####  Clusters")
        st.metric(label="Innovation Clusters", value=selected["clusters"])

    with col2:
        st.markdown("####  Edges")
        st.metric(label="Edges in Graph", value=selected["edges"])

    with col3:
        st.markdown("####  Notes")
        st.markdown(f"<div style='padding: 10px; border-radius: 8px; background-color: #f0f2f6;'>{selected['note']}</div>", unsafe_allow_html=True)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 准备数据
methods = list(cluster_data.keys())
edges = [v["edges"] for v in cluster_data.values()]

# 🎨 使用彩虹渐变色（可选其他 colormap，如 viridis, plasma 等）
cmap = cm.get_cmap('rainbow')
colors = [cmap(i / len(methods)) for i in range(len(methods))]

# 📊 绘图
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(methods, edges, color=colors)

# 设置 Y 轴起始值为 10000
ax.set_ylim(12000, max(edges) + 1000)
ax.set_ylabel("Edge Count")
ax.set_title("Edge Count Comparison Across Clustering Methods")

# X轴文字旋转
plt.xticks(rotation=15, ha='right')
st.pyplot(fig)
