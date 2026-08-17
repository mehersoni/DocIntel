import sys
from pathlib import Path

# Add project root to Python path to resolve imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import json
import yaml
import dotenv
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components

# Load environment variables from .env if present
dotenv.load_dotenv()

def compute_cooccurrence_matrix(approved_cats, chunks):
    """
    Computes a symmetric co-occurrence matrix between approved categories across document chunks.
    """
    cat_names = [c["name"] for c in approved_cats]
    n = len(cat_names)
    matrix = [[0] * n for _ in range(n)]
    
    # Map category to search terms
    cat_terms = {}
    for c in approved_cats:
        terms = [c["name"].lower()]
        if c.get("details"):
            terms.extend([t.lower() for t in c["details"].get("examples", [])[:3]])
            terms.extend([t.lower() for t in c["details"].get("observed_terms", [])[:3]])
        cat_terms[c["name"]] = terms
        
    for chunk in chunks:
        text_lower = chunk.get("text", "").lower()
        present = set()
        for c_name, terms in cat_terms.items():
            if any(t in text_lower for t in terms if len(t) > 2):
                present.add(c_name)
                
        present_list = list(present)
        for i in range(len(present_list)):
            idx_i = cat_names.index(present_list[i])
            for j in range(len(present_list)):
                idx_j = cat_names.index(present_list[j])
                matrix[idx_i][idx_j] += 1
                
    return cat_names, matrix

def render_cooccurrence_heatmap(cat_names, matrix):
    """
    Renders an interactive Plotly Heatmap of category co-occurrences.
    """
    if not cat_names or not matrix:
        st.info("No co-occurrence data available.")
        return
        
    fig = px.imshow(
        matrix,
        x=cat_names,
        y=cat_names,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="Category", y="Category", color="Co-occurrences"),
        text_auto=True
    )
    fig.update_layout(
        title="Domain Co-Occurrence Heatmap Across Document Chunks",
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif", size=13),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

def render_knowledge_graph_html(cat_names, matrix, approved_cats):
    """
    Renders an interactive 2D Visual Knowledge Graph using HTML5 Canvas & Vis.js.
    """
    if not cat_names:
        st.info("No network nodes available.")
        return
        
    nodes_data = []
    for c in approved_cats:
        c_name = c["name"]
        match_count = c.get("match_count", 5)
        nodes_data.append({
            "id": c_name,
            "label": c_name,
            "value": max(match_count, 10),
            "title": f"Category: {c_name}<br>Mentions: {match_count}"
        })
        
    edges_data = []
    for i in range(len(cat_names)):
        for j in range(i + 1, len(cat_names)):
            weight = matrix[i][j]
            if weight > 0:
                edges_data.append({
                    "from": cat_names[i],
                    "to": cat_names[j],
                    "value": weight,
                    "title": f"Co-occurrences: {weight}"
                })
                
    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
      <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
      <style type="text/css">
        #network {{
          width: 100%;
          height: 460px;
          border: 1px solid #E2E8F0;
          background-color: #F8FAFC;
          border-radius: 8px;
        }}
      </style>
    </head>
    <body>
      <div id="network"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          nodes: {{
            shape: 'dot',
            size: 24,
            font: {{ size: 14, face: 'Inter, sans-serif', color: '#0F172A' }},
            color: {{ background: '#3B82F6', border: '#1D4ED8', highlight: {{ background: '#60A5FA', border: '#2563EB' }} }}
          }},
          edges: {{
            color: {{ color: '#94A3B8', highlight: '#3B82F6' }},
            smooth: {{ type: 'continuous' }}
          }},
          physics: {{
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {{ gravitationalConstant: -50, centralGravity: 0.01, springLength: 100, springConstant: 0.08 }}
          }}
        }};
        var network = new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=480)

# App imports
from app.config import (
    UPLOADS_DIR,
    SAMPLE_REPORTS_DIR,
    CHUNKS_PATH,
    SCHEMA_DISCOVERY_INPUT_PATH,
    CANDIDATE_SCHEMA_PATH,
    APPROVED_SCHEMA_PATH,
    EDIT_LOG_PATH,
    DEFAULT_TOP_N
)
from app.ingestion.extractor import extract_text, extract_captions, extract_images
from app.ingestion.chunker import chunk_text, save_chunks
from app.discovery.statistical import run_statistical_discovery
from app.discovery.glossary_scan import run_direct_glossary_scan as run_llm_discovery, extract_category_details
from app.discovery.merger import merge_discovered_categories
from app.schema.manager import load_yaml_schema, save_approved_schema

# --- Custom Page Config & Design ---
st.set_page_config(
    page_title="DocIntel",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Google Fonts and Custom CSS for an official, clean design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0F172A; /* Deep Slate */
        margin-bottom: 0.25rem;
    }
    
    .subtitle {
        font-size: 1.05rem;
        color: #475569; /* Slate 600 */
        margin-bottom: 1.5rem;
    }
    
    .section-card {
        padding: 1.25rem;
        border-radius: 0.375rem;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin-bottom: 0.75rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1.25;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 0.25rem;
    }
    
    .badge-high {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #BCF0DA;
    }
    
    .badge-medium {
        background-color: #E1EFFE;
        color: #1E429F;
        border: 1px solid #C3DDFD;
    }
    
    .badge-source {
        background-color: #F3F4F6;
        color: #374151;
        border: 1px solid #E5E7EB;
    }
    
    .badge-user {
        background-color: #F3E8FF;
        color: #6B21A8;
        border: 1px solid #E9D5FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Session State Initialization ---
if "discovery_completed" not in st.session_state:
    st.session_state.discovery_completed = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "candidate_categories" not in st.session_state:
    st.session_state.candidate_categories = []
if "reviewed_categories" not in st.session_state:
    st.session_state.reviewed_categories = []
if "approved" not in st.session_state:
    st.session_state.approved = False
if "edit_log" not in st.session_state:
    st.session_state.edit_log = {"added": [], "removed": [], "renamed": {}}

# --- Sidebar Configuration ---
st.sidebar.markdown("## Configuration")
st.sidebar.markdown("🔒 **Execution Mode**: 100% Offline (Local)")

# Advanced Settings
st.sidebar.markdown("### Advanced Settings")
top_n = st.sidebar.slider("Top-N Statistical Noun Phrases", min_value=5, max_value=50, value=DEFAULT_TOP_N, step=5, help="Number of noun phrases sent from spaCy to the local glossary for matching.")

api_key = None # Offline Mode (API Key no longer required)

# Reset Button
if st.sidebar.button("Reset Project Space"):
    # Delete outputs on disk
    for path in [CANDIDATE_SCHEMA_PATH, APPROVED_SCHEMA_PATH, EDIT_LOG_PATH, SCHEMA_DISCOVERY_INPUT_PATH, CHUNKS_PATH]:
        if path.exists():
            path.unlink()
    st.session_state.discovery_completed = False
    st.session_state.chunks = []
    st.session_state.candidate_categories = []
    st.session_state.reviewed_categories = []
    st.session_state.approved = False
    st.session_state.edit_log = {"added": [], "removed": [], "renamed": {}}
    st.success("Session state and output files cleared!")
    st.rerun()

# --- Main App Header ---
st.markdown("<h1 class='main-title'>DocIntel</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Human-in-the-loop schema discovery from unstructured technical reports</p>", unsafe_allow_html=True)

# Tabs
tab_ingest, tab_review, tab_glossary = st.tabs(["Ingestion & Discovery", "Review & Export", "Glossary Dictionary"])

# ==================== TAB 1: INGESTION & DISCOVERY ====================
with tab_ingest:
    st.markdown("### 1. Load Documents")
    
    doc_source = st.radio("Choose Document Source", ["Upload Custom Files", "Use Demo Dataset"], horizontal=True)
    
    selected_files = []
    
    if doc_source == "Upload Custom Files":
        uploaded_files = st.file_uploader(
            "Upload technical PDF, DOCX or TXT files", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            for uf in uploaded_files:
                dest_path = UPLOADS_DIR / uf.name
                with open(dest_path, "wb") as f:
                    f.write(uf.getbuffer())
                selected_files.append(dest_path)
    else:
        # Check if demo reports exist
        demo_files = list(SAMPLE_REPORTS_DIR.glob("*"))
        if not demo_files:
            st.warning("No files found in sample_reports directory.")
        else:
            st.markdown("**Select Demo Files to Process:**")
            for df in demo_files:
                if st.checkbox(f"{df.name} (Demo)", value=True, key=f"demo_{df.name}"):
                    selected_files.append(df)
                    
    st.markdown("---")
    
    # Discovery Action
    st.markdown("### 2. Discover Entity Categories")
    
    discover_button = st.button(
        "Discover Schema", 
        type="primary", 
        disabled=not selected_files
    )
    
    if discover_button:
        try:
            # Clean up old outputs to prevent mixing states
            for out_p in [CANDIDATE_SCHEMA_PATH, APPROVED_SCHEMA_PATH, EDIT_LOG_PATH, SCHEMA_DISCOVERY_INPUT_PATH]:
                if out_p.exists():
                    out_p.unlink()
            
            with st.spinner("Step 1: Extracting text, figures, and images..."):
                all_chunks = []
                all_captions = []
                all_images = []
                for file_path in selected_files:
                    doc_id = file_path.name
                    extracted_txt = extract_text(file_path)
                    captions = extract_captions(extracted_txt)
                    all_captions.extend(captions)
                    imgs = extract_images(file_path)
                    all_images.extend(imgs)
                    chunks = chunk_text(extracted_txt, doc_id=doc_id, chunk_size=500)
                    all_chunks.extend(chunks)
                
                if not all_chunks:
                    st.error("No text could be extracted from selected files.")
                    st.stop()
                
                # Save chunks.json
                save_chunks(all_chunks, CHUNKS_PATH)
                st.session_state.chunks = all_chunks
                st.session_state.document_captions = all_captions
                st.session_state.extracted_images = all_images
                
                # Save outputs/captions.json
                captions_path = Path("outputs/captions.json")
                captions_path.parent.mkdir(parents=True, exist_ok=True)
                with open(captions_path, "w", encoding="utf-8") as f:
                    json.dump(all_captions, f, indent=2, ensure_ascii=False)
                    
                # Save outputs/extracted_images.json
                images_json_path = Path("outputs/extracted_images.json")
                with open(images_json_path, "w", encoding="utf-8") as f:
                    json.dump(all_images, f, indent=2, ensure_ascii=False)
                    
                st.info(f"Ingested {len(selected_files)} document(s) and split into {len(all_chunks)} chunks.")
                
            with st.spinner("Step 2: Running spaCy Statistical Discovery & Glossary Matching..."):
                # Run statistical
                stat_cats = run_statistical_discovery(
                    chunks=all_chunks, 
                    api_key=api_key, 
                    top_n=top_n
                )
                st.success(f"Discovered {len(stat_cats)} categories via Statistical method.")
                
            with st.spinner("Step 3: Running Direct Glossary Scan (Offline)..."):
                # Run direct glossary scan
                llm_cats = run_llm_discovery(
                    chunks=all_chunks,
                    api_key=api_key
                )
                st.success(f"Discovered {len(llm_cats)} categories via Direct Glossary Scan (Offline).")
                
            with st.spinner("Step 4: Performing Hybrid Merge and generating candidate_schema.yaml..."):
                merged = merge_discovered_categories(stat_cats, llm_cats, CANDIDATE_SCHEMA_PATH)
                
                st.session_state.candidate_categories = merged
                st.session_state.reviewed_categories = []
                for cat in merged:
                    details = extract_category_details(
                        category_name=cat["name"],
                        category_description=cat["description"],
                        chunks=all_chunks,
                        api_key=api_key
                    )
                    st.session_state.reviewed_categories.append({
                        "name": cat["name"],
                        "original_name": cat["name"],
                        "description": cat["description"],
                        "confidence": cat["confidence"],
                        "sources": cat["sources"],
                        "approved": True,
                        "is_new": False,
                        "match_count": cat.get("match_count", 0),
                        "details": details
                    })
                
                st.session_state.discovery_completed = True
                st.session_state.approved = False
                st.session_state.edit_log = {"added": [], "removed": [], "renamed": {}}
                
                st.success("Schema discovery completed! Head over to the 'Review & Export' tab.")
                
        except Exception as e:
            st.error(f"Error during discovery: {e}")

# ==================== TAB 2: REVIEW & EXPORT ====================
with tab_review:
    if not st.session_state.discovery_completed:
        st.info("Please upload/select documents and click 'Discover Schema' in the Ingestion & Discovery tab first.")
    else:
        # --- Executive KPI Metrics Bar ---
        approved_cats_list = [c for c in st.session_state.reviewed_categories if c.get("approved", True)]
        total_term_mentions = sum(c.get("match_count", 0) for c in approved_cats_list)
        high_conf_count = sum(1 for c in approved_cats_list if c.get("confidence") == "HIGH")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(label="📄 Text Chunks Parsed", value=len(st.session_state.chunks))
        with kpi2:
            st.metric(label="🔍 Discovered Categories", value=len(st.session_state.reviewed_categories))
        with kpi3:
            st.metric(label="⭐ High-Confidence Disciplines", value=high_conf_count)
        with kpi4:
            st.metric(label="🎯 Technical Term Mentions", value=total_term_mentions)
            
        st.markdown("---")

        # Sub-tabs for Review & Export workspace
        subtab_review, subtab_analytics, subtab_export = st.tabs([
            "📋 Category Review & Approval", 
            "📊 Executive Synthesis & Analytics", 
            "💾 Schema Approval & Downloads"
        ])

        # ------------ SUBTAB 1: CATEGORY REVIEW & APPROVAL ------------
        with subtab_review:
            st.markdown("#### Category Schema Review")
            st.caption("Toggle checkboxes to include/reject, edit names or descriptions, or expand details to customize terms and measurements.")
            
            # Add Custom Category Form in an expander
            with st.expander("➕ Add Custom Category", expanded=False):
                with st.form("add_cat_form", clear_on_submit=True):
                    new_name = st.text_input("Category Name (CamelCase, e.g. Contractor)", placeholder="Contractor")
                    new_desc = st.text_area("Category Description", placeholder="Drilling or service contractors...")
                    submit_add = st.form_submit_button("Add Category")
                    
                    if submit_add:
                        clean_new_name = new_name.strip()
                        if not clean_new_name:
                            st.error("Category Name cannot be empty.")
                        elif any(c["name"].lower() == clean_new_name.lower() for c in st.session_state.reviewed_categories if c["approved"]):
                            st.error("A category with this name already exists.")
                        else:
                            st.session_state.reviewed_categories.append({
                                "name": clean_new_name,
                                "original_name": clean_new_name,
                                "description": new_desc.strip(),
                                "confidence": "HIGH",
                                "sources": ["user"],
                                "approved": True,
                                "is_new": True,
                                "match_count": 0
                            })
                            st.success(f"Category '{clean_new_name}' added successfully!")
                            st.rerun()

            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
            
            # Display candidate categories in clean collapsed expander cards
            to_delete = []
            for idx, cat in enumerate(st.session_state.reviewed_categories):
                conf = cat["confidence"]
                match_count = cat.get("match_count", 0)
                status_icon = "✅" if cat["approved"] else "❌"
                
                expander_label = f"{status_icon} {cat['name']} — {conf} Confidence ({match_count} mentions)"
                with st.expander(expander_label, expanded=False):
                    col_check, col_edit, col_desc, col_del = st.columns([1, 3, 5, 1])
                    
                    with col_check:
                        cat["approved"] = st.checkbox("Include", value=cat["approved"], key=f"check_{idx}")
                    with col_edit:
                        cat["name"] = st.text_input("Name", value=cat["name"], key=f"name_{idx}", disabled=not cat["approved"])
                    with col_desc:
                        cat["description"] = st.text_input("Description", value=cat["description"], key=f"desc_{idx}", disabled=not cat["approved"])
                    with col_del:
                        if st.button("Delete", key=f"del_{idx}", help="Remove category"):
                            to_delete.append(idx)

                    # Manage category details inside nested sub-tabs for clean layout
                    if cat["approved"]:
                        st.markdown("---")
                        
                        # Enrichment trigger
                        btn_lbl = "Regenerate Enriched Details" if cat.get("details") else "Extract Enriched Details"
                        if st.button(btn_lbl, key=f"extract_btn_{idx}"):
                            with st.spinner(f"Extracting details for '{cat['name']}'..."):
                                try:
                                    from app.discovery.llm import extract_category_details
                                    from app.schema.manager import merge_details
                                    new_d = extract_category_details(
                                        category_name=cat["name"],
                                        category_description=cat["description"],
                                        chunks=st.session_state.chunks
                                    )
                                    cat["details"] = merge_details(cat.get("details", {}), new_d)
                                    st.success(f"Enriched details for '{cat['name']}'.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to extract details: {e}")
                                    
                        if cat.get("details"):
                            dt1, dt2 = st.tabs(["📝 Summary & Terms", "💬 Mentions & Measurements"])
                            
                            with dt1:
                                cat["details"]["summary"] = st.text_input("Category Summary", value=cat["details"].get("summary", ""), key=f"summary_val_{idx}")
                                
                                col_ex, col_terms = st.columns(2)
                                with col_ex:
                                    st.markdown("**Representative Examples**")
                                    ex_str = ", ".join(cat["details"].get("examples", []))
                                    new_ex_str = st.text_area("Examples (comma-separated)", value=ex_str, key=f"ex_area_{idx}", height=75)
                                    cat["details"]["examples"] = [x.strip() for x in new_ex_str.split(",") if x.strip()]
                                    
                                with col_terms:
                                    st.markdown("**Observed Terms**")
                                    terms_str = ", ".join(cat["details"].get("observed_terms", []))
                                    new_terms_str = st.text_area("Terms (comma-separated)", value=terms_str, key=f"terms_area_{idx}", height=75)
                                    cat["details"]["observed_terms"] = [t.strip() for t in new_terms_str.split(",") if t.strip()]

                            with dt2:
                                st.markdown("**Sentence Mentions**")
                                mentions = cat["details"].get("sample_mentions", [])
                                for m in mentions[:3]:
                                    if "No specific mentions" not in m:
                                        st.markdown(f"> {m}")
                                        
                                num_attrs = cat["details"].get("numerical_attributes", [])
                                if num_attrs:
                                    st.markdown("**Extracted Numerical Measurements**")
                                    num_tags = " ".join([f"<span class='badge badge-medium'>{attr}</span>" for attr in num_attrs])
                                    st.markdown(num_tags, unsafe_allow_html=True)
                                    
            if to_delete:
                for index in sorted(to_delete, reverse=True):
                    st.session_state.reviewed_categories.pop(index)
                st.rerun()

        # ------------ SUBTAB 2: EXECUTIVE SYNTHESIS & ANALYTICS ------------
        with subtab_analytics:
            approved_cats = [c for c in st.session_state.reviewed_categories if c.get("approved", True)]
            approved_cats.sort(key=lambda x: x.get("match_count", 0), reverse=True)
            total_matches = sum(c.get("match_count", 0) for c in approved_cats)
            
            st.markdown("#### Executive Technical Synthesis")
            if approved_cats:
                top_cat = approved_cats[0]
                top_count = top_cat.get("match_count", 0)
                top_pct = f"{(top_count / total_matches * 100):.1f}%" if total_matches > 0 else "0%"
                
                overview_para = (
                    f"The analyzed document corpus encompasses **{len(approved_cats)}** approved technical entity categories "
                    f"with a total of **{total_matches}** term occurrences. "
                    f"The primary technical domain is **{top_cat['name']}**, representing **{top_count}** mentions "
                    f"({top_pct} of total identified technical terminology)."
                )
                
                if len(approved_cats) > 1:
                    sec_cats = approved_cats[1:3]
                    sec_str = ", ".join([f"**{c['name']}** ({c.get('match_count', 0)} mentions)" for c in sec_cats])
                    top3_count = sum(c.get("match_count", 0) for c in approved_cats[:3])
                    top3_pct = f"{(top3_count / total_matches * 100):.1f}%" if total_matches > 0 else "0%"
                    overview_para += f" Secondary focus areas include {sec_str}. Together, these top domains account for **{top3_pct}** of all domain mentions in the corpus."
                
                st.info(overview_para)
                
                st.markdown("#### Key Domain Breakdown")
                for c in approved_cats[:4]:
                    match_cnt = c.get("match_count", 0)
                    pct_str = f"{(match_cnt / total_matches * 100):.1f}%" if total_matches > 0 else "0%"
                    c_summary = c.get("details", {}).get("summary", "") if c.get("details") else ""
                    if not c_summary or "no direct mentions" in c_summary.lower():
                        examples = c.get("details", {}).get("examples", [])[:3] if c.get("details") else []
                        ex_str = ", ".join(examples) if examples else "technical concepts"
                        c_summary = f"Incorporates terminology and operational concepts related to {c['name'].lower()}, featuring items such as {ex_str}."
                    st.markdown(f"- **{c['name']}** *(Volume: {match_cnt} mentions | {pct_str} share)*: {c_summary}")
                    
                summary_for_export = overview_para.replace("**", "") + "\n\nKey Domain Insights:\n"
                for c in approved_cats[:4]:
                    c_summary = c.get("details", {}).get("summary", "") if c.get("details") else ""
                    if not c_summary or "no direct mentions" in c_summary.lower():
                        examples = c.get("details", {}).get("examples", [])[:3] if c.get("details") else []
                        ex_str = ", ".join(examples) if examples else "technical concepts"
                        c_summary = f"Incorporates terminology and operational concepts related to {c['name'].lower()}, featuring items such as {ex_str}."
                    summary_for_export += f"- {c['name']}: {c_summary}\n"
                st.session_state.document_summary = summary_for_export
            else:
                st.warning("No entity categories are approved. Please approve categories in the review tab.")

            st.markdown("---")
            
            # Interactive Visualizations
            st.markdown("#### Domain Co-Occurrence & Knowledge Graph")
            if approved_cats and st.session_state.chunks:
                cat_names, cooc_matrix = compute_cooccurrence_matrix(approved_cats, st.session_state.chunks)
                vtab1, vtab2 = st.tabs(["Co-Occurrence Heatmap", "Interactive Knowledge Graph"])
                with vtab1:
                    render_cooccurrence_heatmap(cat_names, cooc_matrix)
                with vtab2:
                    render_knowledge_graph_html(cat_names, cooc_matrix, approved_cats)

            # Figures & Extracted Document Images
            images_json_path = Path("outputs/extracted_images.json")
            extracted_imgs = []
            if images_json_path.exists():
                with open(images_json_path, "r", encoding="utf-8") as f:
                    extracted_imgs = json.load(f)
            elif st.session_state.get("extracted_images"):
                extracted_imgs = st.session_state.extracted_images

            captions_path = Path("outputs/captions.json")
            captions = []
            if captions_path.exists():
                with open(captions_path, "r", encoding="utf-8") as f:
                    captions = json.load(f)
            elif st.session_state.get("document_captions"):
                captions = st.session_state.document_captions

            if extracted_imgs or captions:
                st.markdown("#### 🖼️ Document Figures, Diagrams & Illustrations")
                
                # Display extracted images visually if available
                valid_imgs = [img for img in extracted_imgs if Path(img["path"]).exists()]
                if valid_imgs:
                    img_cols = st.columns(min(len(valid_imgs), 3))
                    for i_idx, img_info in enumerate(valid_imgs):
                        col = img_cols[i_idx % len(img_cols)]
                        with col:
                            st.image(
                                img_info["path"],
                                caption=img_info.get("caption") or f"Figure {i_idx+1}",
                                use_container_width=True
                            )
                            
                # Display text figure captions
                if captions:
                    for cap in captions:
                        st.markdown(f" * **Figure/Diagram**: {cap}")

        # ------------ SUBTAB 3: APPROVE & DOWNLOAD SCHEMA ------------
        with subtab_export:
            st.markdown("#### Finalize Schema Approval")
            st.write("Click 'Approve Schema' below to commit user edits and generate final schema YAML and audit edit logs.")
            
            if st.button("Approve Schema", type="primary"):
                valid = True
                seen_names = set()
                for cat in st.session_state.reviewed_categories:
                    if cat["approved"]:
                        name = cat["name"].strip()
                        if not name:
                            st.error("Approved category names cannot be empty.")
                            valid = False
                            break
                        if name.lower() in seen_names:
                            st.error(f"Duplicate approved category name detected: '{name}'")
                            valid = False
                            break
                        seen_names.add(name.lower())
                        
                if valid:
                    added = []
                    removed = []
                    renamed = {}
                    approved_cats = []
                    
                    for cat in st.session_state.reviewed_categories:
                        if cat["approved"]:
                            approved_item = {
                                "name": cat["name"].strip(),
                                "description": cat["description"].strip(),
                                "confidence": cat["confidence"],
                                "sources": cat["sources"]
                            }
                            if cat.get("details"):
                                approved_item["details"] = cat["details"]
                            approved_cats.append(approved_item)
                            
                            if cat["is_new"]:
                                added.append(cat["name"].strip())
                            elif cat["name"].strip() != cat["original_name"]:
                                renamed[cat["original_name"]] = cat["name"].strip()
                                
                    for c in st.session_state.candidate_categories:
                        orig_name = c["name"]
                        is_active = any(cat["approved"] and cat["original_name"] == orig_name for cat in st.session_state.reviewed_categories)
                        if not is_active:
                            removed.append(orig_name)
                            
                    st.session_state.edit_log = {"added": added, "removed": removed, "renamed": renamed}
                    
                    # Load extracted images for export
                    images_json_path = Path("outputs/extracted_images.json")
                    extracted_imgs = []
                    if images_json_path.exists():
                        with open(images_json_path, "r", encoding="utf-8") as f:
                            extracted_imgs = json.load(f)
                    elif st.session_state.get("extracted_images"):
                        extracted_imgs = st.session_state.extracted_images

                    save_approved_schema(
                        approved_categories=approved_cats,
                        edit_log=st.session_state.edit_log,
                        approved_path=APPROVED_SCHEMA_PATH,
                        edit_log_path=EDIT_LOG_PATH,
                        document_summary=st.session_state.get("document_summary", ""),
                        document_captions=st.session_state.get("document_captions", []),
                        extracted_images=extracted_imgs
                    )
                    st.session_state.approved = True
                    st.success("Approved Schema and Edit Log successfully written to outputs directory!")

            st.markdown("---")
            st.markdown("#### Downloadable Artifacts & Output Previews")
            
            cand_yaml = CANDIDATE_SCHEMA_PATH.read_text(encoding="utf-8") if CANDIDATE_SCHEMA_PATH.exists() else ""
            appr_yaml = APPROVED_SCHEMA_PATH.read_text(encoding="utf-8") if APPROVED_SCHEMA_PATH.exists() else ""
            edit_json_str = EDIT_LOG_PATH.read_text(encoding="utf-8") if EDIT_LOG_PATH.exists() else ""
            input_json_str = SCHEMA_DISCOVERY_INPUT_PATH.read_text(encoding="utf-8") if SCHEMA_DISCOVERY_INPUT_PATH.exists() else ""
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if cand_yaml:
                    st.download_button("Download Candidate Schema (YAML)", data=cand_yaml, file_name="candidate_schema.yaml", mime="text/yaml", use_container_width=True)
            with col2:
                if appr_yaml:
                    st.download_button("Download Approved Schema (YAML)", data=appr_yaml, file_name="approved_schema.yaml", mime="text/yaml", use_container_width=True)
            with col3:
                if edit_json_str:
                    st.download_button("Download Edit Log (JSON)", data=edit_json_str, file_name="edit_log.json", mime="application/json", use_container_width=True)
            with col4:
                if input_json_str:
                    st.download_button("Download Discovery Input (JSON)", data=input_json_str, file_name="schema_discovery_input.json", mime="application/json", use_container_width=True)

            if appr_yaml or edit_json_str:
                st.markdown("##### Previews")
                
                # Render extracted document images inside final summary preview if present
                images_json_path = Path("outputs/extracted_images.json")
                final_imgs = []
                if images_json_path.exists():
                    with open(images_json_path, "r", encoding="utf-8") as f:
                        final_imgs = json.load(f)
                elif st.session_state.get("extracted_images"):
                    final_imgs = st.session_state.extracted_images

                valid_final_imgs = [img for img in final_imgs if Path(img["path"]).exists()]
                if valid_final_imgs:
                    st.markdown("###### 🖼️ Extracted Figures Displayed in Final Summary")
                    f_cols = st.columns(min(len(valid_final_imgs), 3))
                    for idx, img_info in enumerate(valid_final_imgs):
                        col = f_cols[idx % len(f_cols)]
                        with col:
                            st.image(
                                img_info["path"],
                                caption=img_info.get("caption") or f"Final Summary Figure {idx+1}",
                                use_container_width=True
                            )
                    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    if appr_yaml:
                        st.markdown("**Approved Schema (YAML)**")
                        st.code(appr_yaml, language="yaml")
                with pcol2:
                    if edit_json_str:
                        st.markdown("**Edit Log (JSON)**")
                        st.code(edit_json_str, language="json")

# ==================== TAB 3: GLOSSARY DICTIONARY ====================
with tab_glossary:
    st.markdown("### Browse the Local Geology & Oilfield Glossary")
    
    glossary_path = os.path.join("data", "oilfield_glossary.json")
    if not os.path.exists(glossary_path):
        st.info("Glossary file data/oilfield_glossary.json does not exist. Run generate_massive_glossary.py first.")
    else:
        with open(glossary_path, "r", encoding="utf-8") as f:
            glossary = json.load(f)
            
        # Get unique categories
        categories = set()
        for entry in glossary.values():
            categories.add(entry["category"])
        sorted_cats = sorted(list(categories))
        
        # Display stat
        st.markdown(f"<div style='margin-bottom: 1rem;'>Loaded <b>{len(glossary):,}</b> technical terms across <b>{len(sorted_cats)}</b> categories.</div>", unsafe_allow_html=True)
        
        # Filters row
        col_search, col_cat = st.columns([2, 1])
        with col_search:
            search_query = st.text_input("Search Terms or Descriptions", placeholder="Search for shale, seismic, packer, gamma...", key="glossary_search")
        with col_cat:
            selected_cat = st.selectbox("Category Filter", ["All Categories"] + sorted_cats, key="glossary_cat")
            
        # Perform filtering
        filtered_terms = []
        for key, entry in glossary.items():
            # Apply category filter
            if selected_cat != "All Categories" and entry["category"] != selected_cat:
                continue
            # Apply search filter
            if search_query:
                q = search_query.lower()
                in_term = q in entry["term"].lower()
                in_desc = q in entry["description"].lower()
                in_syns = any(q in syn.lower() for syn in entry.get("synonyms", []))
                if not (in_term or in_desc or in_syns):
                    continue
            filtered_terms.append(entry)
            
        # Sorting matches
        filtered_terms.sort(key=lambda x: x["term"])
        
        # Display matches
        total_matches = len(filtered_terms)
        max_display = 100
        
        st.markdown(f"**Showing {min(max_display, total_matches)} of {total_matches:,} matches:**")
        
        for idx, entry in enumerate(filtered_terms[:max_display]):
            term_name = entry["term"]
            cat_label = entry["category"]
            desc_text = entry["description"]
            attributes = entry.get("attributes", [])
            syns = entry.get("synonyms", [])
            
            with st.expander(f"{term_name} — {cat_label}", expanded=False):
                st.write(f"**Description:** {desc_text}")
                
                col_syn, col_attr = st.columns(2)
                with col_syn:
                    if syns:
                        st.write("**Synonyms / Aliases:**")
                        st.write(", ".join([s for s in syns if s.lower() != term_name.lower()]))
                with col_attr:
                    if attributes:
                        st.write("**Common Attributes:**")
                        st.write(", ".join(attributes))

