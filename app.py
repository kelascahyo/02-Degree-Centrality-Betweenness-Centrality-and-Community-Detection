import streamlit as st
import pandas as pd
import networkx as nx
import json
import streamlit.components.v1 as components

# Set wide page layout
st.set_page_config(page_title="Tax Risk Network Analytics", layout="wide")

# ----------------------------------------
# 1. TITLE & INTRODUCTORY NARRATIVE
# ----------------------------------------
st.title("🕸️ Tax Risk Profiling & Network Analytics Dashboard")
st.markdown("""
### Executive Summary & Methodology
This interactive web application leverages **Graph Theory (Network Analytics)** to uncover hidden patterns, ownership clusters, and tax risk profiles within corporate networks. 
Instead of looking at tax entities in isolation, this tool models the tax data as a **Directed Network Graph**:
* **Nodes (Vertices):** Represent Taxpayers (Companies, Foreign Entities, Individuals, or Unregistered/Non-NPWP entities).
* **Edges (Links):** Represent Shareholding Ownership structures, carrying financial weights like ownership percentage, absolute share value, and paid dividends.

By using **NetworkX** for backend algorithmic processing and **D3.js** for frontend web-based visualization, we pinpoint highly critical nodes that could act as systematic tax risks.
""")

# ----------------------------------------
# 2. DATA LOADING & CACHING
# ----------------------------------------
@st.cache_data
def load_data():
    # Load masked data
    nodes_df = pd.read_csv("nodes_masked.csv")
    edges_df = pd.read_csv("edges_masked.csv")
    return nodes_df, edges_df

try:
    nodes_df, edges_df = load_data()
except Exception as e:
    st.error(f"Error loading CSV files. Please make sure 'nodes_masked.csv' and 'edges_masked.csv' are in the root directory. Error: {e}")
    st.stop()

# Sidebar controls for filtering data to prevent web browser crashing on huge networks
st.sidebar.header("🔍 Network Filter Controls")
st.sidebar.markdown("Given the large volume of corporate connections, filter the network to focus on high-value or highly-connected clusters.")

min_percentage = st.sidebar.slider("Minimum Shareholding Percentage (%)", 0.0, 100.0, 1.0, 0.5)
min_value = st.sidebar.number_input("Minimum Share Value (IDR)", min_value=0, value=10_000_000, step=5_000_000)

# Filter edges
filtered_edges = edges_df[
    (edges_df['persentase'] >= min_percentage) & 
    (edges_df['nilai'] >= min_value)
]

# Get active nodes involved in filtered edges
active_node_ids = set(filtered_edges['sumber']).union(set(filtered_edges['target']))
filtered_nodes = nodes_df[nodes_df['id'].isin(active_node_ids)].copy()

# ----------------------------------------
# 3. BACKEND PROCESSING (NetworkX)
# ----------------------------------------
# Build Directed Graph
G = nx.DiGraph()

# Add nodes with attributes
for _, row in filtered_nodes.iterrows():
    G.add_node(row['id'], name=str(row['nama']), type=str(row['jenis_node']))

# Add edges with attributes
for _, row in filtered_edges.iterrows():
    G.add_edge(
        row['sumber'], 
        row['target'], 
        rel_id=int(row['rel_id']),
        percentage=float(row['persentase']),
        value=float(row['nilai']),
        dividend=float(row['dividen'])
    )

# Calculate Network Metrics
# 1. Out-Degree Centrality (Investment Expansion / Holding Company Identification)
out_degree = dict(G.out_degree())
max_out = max(out_degree.values()) if out_degree else 1

# 2. Betweenness Centrality (Intermediary / Conduit / Risk-bridge Identification)
# We treat it as undirected for structural bridging or keep directed for flow. Let's use standard directed betweenness.
betweenness = nx.betweenness_centrality(G)
max_bet = max(betweenness.values()) if betweenness else 1

# 3. Community Detection (Louvain Algorithm)
# Louvain requires an undirected graph
G_undirected = G.to_undirected()
try:
    # Available in NetworkX 3.x+
    communities = nx.community.louvain_communities(G_undirected, seed=42)
    # Map node to community index
    community_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            community_map[node] = idx
except Exception:
    # Fallback if algorithm fails or graph is empty
    community_map = {node: 0 for node in G.nodes()}

# Attach computed metrics back to nodes for D3 ingestion
d3_nodes = []
for node_id in G.nodes():
    node_attr = G.nodes[node_id]
    d3_nodes.append({
        "id": str(node_id),
        "name": node_attr.get("name", "Unknown"),
        "type": node_attr.get("type", "Unknown"),
        "out_degree": int(out_degree.get(node_id, 0)),
        "betweenness": float(betweenness.get(node_id, 0.0)),
        "community": int(community_map.get(node_id, 0))
    })

d3_edges = []
for u, v, data in G.edges(data=True):
    d3_edges.append({
        "source": str(u),
        "target": str(v),
        "percentage": data["percentage"],
        "value": data["value"],
        "dividend": data["dividend"]
    })

# Combine into D3 JSON format
network_json = {"nodes": d3_nodes, "links": d3_edges}

# ----------------------------------------
# 4. DASHBOARD TABS & RISK NARRATIVES
# ----------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Network Insights & Metrics", "🌐 Interactive D3.js Visualizer", "🚨 Risk Profile Interpretation"])

with tab1:
    st.subheader("Key Network Risk Indicators")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Analyzed Active Taxpayers", f"{len(G.nodes()):,}")
    with col2:
        st.metric("Analyzed Ownership Relations", f"{len(G.edges()):,}")
    with col3:
        st.metric("Detected Conglomeration Groups", f"{len(set(community_map.values())):,}")

    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("""
        #### 🏢 Top Potential Holding Companies (High Out-Degree)
        Taxpayers with a **High Out-Degree** act as significant capital providers, spreading investments across multiple subsidiaries. 
        In tax audits, these holding entities are critical because they dictate group-wide transaction policies and asset distributions.
        """)
        top_holding = sorted(d3_nodes, key=lambda x: x['out_degree'], reverse=True)[:5]
        df_holding = pd.DataFrame(top_holding)[['id', 'name', 'type', 'out_degree']]
        st.dataframe(df_holding, use_container_width=True)

    with col_right:
        st.markdown("""
        #### 🔀 Top Strategic Intermediaries (High Betweenness Centrality)
        Taxpayers with **High Betweenness Centrality** control the structural flow of capital between otherwise disconnected groups. 
        **Tax Risk Impact:** These entities frequently serve as conduit companies, shell arrangements, or invoice/dividend bridges designed to exploit transfer pricing and shift profits into low-risk environments.
        """)
        top_bridge = sorted(d3_nodes, key=lambda x: x['betweenness'], reverse=True)[:5]
        df_bridge = pd.DataFrame(top_bridge)[['id', 'name', 'type', 'betweenness']]
        st.dataframe(df_bridge, use_container_width=True)

with tab2:
    st.subheader("Interactive Graph Exploration")
    
    # Selection for Node Size Mapping
    size_mapping = st.radio(
        "Map Node Size (Radius) to Metric:",
        ("High Out-Degree (Holding Company Risk)", "High Betweenness (Conduit/Intermediary Risk)"),
        horizontal=True
    )
    
    # Map selection to D3 data property
    size_prop = "out_degree" if "Out-Degree" in size_mapping else "betweenness"
    
    st.markdown("""
    * **Color-Coding:** Automatically grouped into **Conglomeration Clusters (Louvain Communities)**. Nodes sharing the same color belong to the same tightly knit corporate network.
    * **Interaction:** Drag nodes to rearrange. Hover over a node to reveal full taxpayer details. Scroll to zoom, drag background to pan.
    """)

    # ----------------------------------------
    # 5. D3.JS EMBEDDED JAVASCRIPT & HTML
    # ----------------------------------------
    json_data_str = json.dumps(network_json)
    
    d3_html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Tax Network Visualization</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #fafafa;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                overflow: hidden;
            }}
            #graph-container {{
                width: 100vw;
                height: 650px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background: #ffffff;
            }}
            .links line {{
                stroke: #999;
                stroke-opacity: 0.4;
                stroke-width: 1.5px;
            }}
            .nodes circle {{
                stroke: #fff;
                stroke-width: 1.5px;
                cursor: pointer;
            }}
            .tooltip {{
                position: absolute;
                padding: 10px;
                background: rgba(38, 39, 48, 0.95);
                color: #fff;
                border-radius: 5px;
                font-size: 12px;
                pointer-events: none;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
                line-height: 1.5em;
                display: none;
                z-index: 10;
            }}
            #legend {{
                position: absolute;
                top: 15px;
                left: 15px;
                background: rgba(255,255,255,0.9);
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                font-size: 11px;
                max-height: 150px;
                overflow-y: auto;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin-bottom: 4px;
            }}
            .legend-color {{
                width: 12px;
                height: 12px;
                margin-right: 6px;
                border-radius: 50%;
            }}
        </style>
    </head>
    <body>

        <div id="graph-container"></div>
        <div class="tooltip" id="tooltip"></div>
        
        <script>
            // Parse network data from Streamlit
            const graphData = {json_data_str};
            const sizeProperty = "{size_prop}";

            const width = window.innerWidth;
            const height = 650;

            const svg = d3.select("#graph-container")
                .append("svg")
                .attr("width", "100%")
                .attr("height", height)
                .call(d3.zoom().on("zoom", function (event) {{
                    g.attr("transform", event.transform);
                }}))
                .append("g");

            // Main wrapper group to allow zooming
            const g = svg.append("g");

            // Color palette for communities
            const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

            // Simulation setups
            const simulation = d3.forceSimulation(graphData.nodes)
                .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-120))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(d => getRadius(d) + 5));

            // Tooltip selection
            const tooltip = d3.select("#tooltip");

            // Build Arrowheads for directed connections
            svg.append("defs").flatMap(d => d ? [d] : []).data(["arrow"])
                .enter().append("marker")
                .attr("id", "arrow")
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 23) // offset to position arrow at node boundary
                .attr("refY", 0)
                .attr("markerWidth", 6)
                .attr("markerHeight", 6)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .attr("fill", "#999");

            // Draw links
            const link = g.append("g")
                .attr("class", "links")
                .selectAll("line")
                .data(graphData.links)
                .enter().append("line")
                .attr("marker-end", "url(#arrow)");

            // Dynamic size calculator
            function getRadius(d) {{
                if (sizeProperty === "out_degree") {{
                    return 5 + (d.out_degree * 4);
                }} else {{
                    return 5 + (d.betweenness * 40);
                }}
            }}

            // Draw nodes
            const node = g.append("g")
                .attr("class", "nodes")
                .selectAll("circle")
                .data(graphData.nodes)
                .enter().append("circle")
                .attr("r", d => getRadius(d))
                .attr("fill", d => colorScale(d.community))
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            // Tooltip interactivity
            node.on("mouseover", function(event, d) {{
                tooltip.style("display", "block")
                    .html(`
                        <strong>Taxpayer ID:</strong> ${{d.id}}<br/>
                        <strong>Name:</strong> ${{d.name}}<br/>
                        <strong>Type:</strong> ${{d.type}}<br/>
                        <strong>Group/Community:</strong> Cluster #${{d.community}}<br/>
                        <strong>Out-Degree:</strong> ${{d.out_degree}} subsidiaries<br/>
                        <strong>Betweenness:</strong> ${{d.betweenness.toFixed(4)}}
                    `);
            }})
            .on("mousemove", function(event) {{
                tooltip.style("left", (event.pageX + 15) + "px")
                       .style("top", (event.pageY - 15) + "px");
            }})
            .on("mouseout", function() {{
                tooltip.style("display", "none");
            }});

            // Simulation ticker
            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
            }});

            // Drag behavior handlers
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}

            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}

            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}
        </script>
    </body>
    </html>
    """
    
    # Render the D3 structure into Streamlit
    components.html(d3_html_code, height=660, scrolling=False)

with tab3:
    st.subheader("📚 Audit & Risk Interpretation Guide")
    st.markdown("""
    ### How to Action These Findings for Tax Audits

    #### 1. Understanding Node Diameters (The Ring of Risk)
    * When **High Out-Degree** is selected, nodes that expand outwards with huge circles are **Ultimate Parents / Holding Companies**. Auditing them means looking for consolidated cash flow anomalies or transfer pricing policy directives.
    * When **High Betweenness Centrality** is selected, large circles represent **Conduits**. These entities are extremely high-risk because they connect unrelated corporate clusters. Auditors must verify whether these intermediaries possess true economic substance (employees, physical offices) or are mere *paper companies* designed to pass dividends without triggering fair tax withholding.

    #### 2. Color-Coding & Community Detection (Affiliation Groups)
    * Tax evasion schemes (such as *dividend stripping* or *circular invoicing*) rarely happen inside a single company; they happen across **networks of affiliates**.
    * The automated **Louvain algorithm** detects these hidden networks purely from investment footprints. If an auditor uncovers a tax violation in a **Blue Node**, they should cross-check all other **Blue Nodes** in that same cluster, as corporate policies and financial pipelines are shared within these communities.
    """)
