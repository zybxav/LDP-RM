# Relation Mining Under Local Differential Privacy
This repository is the implementation of the paper:
[Relation Mining Under Local Differential Privacy](https://www.usenix.org/system/files/usenixsecurity24-dong-kai.pdf)
accepted at USENIX Security 2024.  
The experiments on mining relations between items are now available.   
Additional experiments, including those on mining relations among items, mining association rules, and mining items in large domains, as well as other baseline methods and ablation studies, will be continuously updated.
## Requriments
The code is implemented in Python 3.9. Refer to 'requirements.txt' to see all packages.
## Usage
To reproduce the experiments, run `ldp_rm.py` and modify the parameters `epsilon`, `top_k`, `top_ks`, and `top_kc`.  
The datasets currently include **IFTTT (2 items)** and the **MOVIE**, where each row in the dataset represents relations owned by a user.
## Citation 
Please cite our paper as follows:
'''
@inproceedings{dong2024relation,
  title={Relation Mining Under Local Differential Privacy},
  author={Dong, Kai and Zhang, Zheng and Jia, Chuang and Ling, Zhen and Yang, Ming and Luo, Junzhou and Fu, Xinwen},
  booktitle={33rd USENIX Security Symposium (USENIX Security 24)},
  pages={955--972},
  year={2024}
}
'''
