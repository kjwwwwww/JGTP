# JGTP
Training-Free Graph Condensation via Joint-Granularity Topology Preservation
<img width="1536" height="1024" alt="a60a9281-2e16-4d5e-916d-c224b4e9f5f7" src="https://github.com/user-attachments/assets/ea1eb190-6845-4e09-b393-b47ee6c3b792" />
## Download Datasets

For Cora, Citeseer, and Pubmed, the code will directly download them from PyTorch Geometric.  
For Flickr, Ogbn-arxiv, and Reddit, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT).  
They are available on [this Google Drive link](#) provided by the GraphSAINT team.  
Download the files and unzip them to `datasets` at the root directory.
| Dataset | #Nodes | #Edges | #Classes | #Features |
|---|---:|---:|---:|---:|
| Cora | 2,708 | 10,556 | 7 | 1,433 |
| Citeseer | 3,327 | 9,104 | 6 | 3,703 |
| Pubmed | 19,717 | 88,648 | 3 | 500 |
| Flickr | 89,250 | 899,756 | 7 | 500 |
| Ogbn-arxiv | 169,343 | 2,315,598 | 40 | 128 |
| Reddit | 232,965 | 23,213,838 | 41 | 602 |
| MAG240M | 1,398,159 | 26,434,726 | 153 | 768 |
