import pandas as pd
from nichespace.manifold import QualitativeSpace

X_basis = pd.read_parquet("../data/manifold/v2025.3.3/completeness_gte50.contamination_lt10/NAL-GDB_MNS__v2025.3.3__SLC-MFC.medium.HierarchicalNicheSpace.X_basis.parquet")
y_basis = pd.read_parquet("../data/manifold/v2025.3.3/completeness_gte50.contamination_lt10/NAL-GDB_MNS__v2025.3.3__SLC-MFC.medium.HierarchicalNicheSpace.y_basis.parquet").squeeze()
qualitative_mns = QualitativeSpace(
    observation_type="genome",
    feature_type="ko",
    class_type="mfc-cluster",
    name="NAL-GDB_MNS__v2025.3.3__SLC-MFC.medium",
    n_trials=50,
    verbose=0,
    n_neighbors=19, 
    n_components=3,
)
qualitative_mns.fit(X_basis, y_basis)
qualitative_mns.to_file("../data/manifold/v2025.3.3/completeness_gte50.contamination_lt10/NAL-GDB_MNS__v2025.3.3__SLC-MFC.medium.QualitativeSpace.pkl")
