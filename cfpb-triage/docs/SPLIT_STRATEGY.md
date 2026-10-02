# Split Strategy
TODO (Subangan)
Must focus on order to account for leakage
1. Order of operations 
    - Normalize text and assign group_id (exact + near duplicate matching)
    - Colapse to one row per group(majoriy label,group_size retained)
    - Drop calsses wit hfewer than 200 rows, counted after collapsing since collapsing shrinks class counts
    - Split into Train,validation and test models
    -fit all preprocessing (TF-IDF vocabulary,IDF weights ,label encoder,naive bayes counts on train only)
**Collapsing and class filtering happen before the split so that class counts and stratification reflect the final modelling set.**
2. Split Design
    - Ratio goal : 70% train, 15% Validation and 15% Test
    - Stratification: By Product, so each split will keep roughly the same class proportions. With the 200 row minimum the smallest cass has at least about 30 test rows. Which will be enough for a stable per class estimation 
3. Roles of Each Split
    - Train : fit the cevtorizer and models
    - Validation : Tune Hyperparameters( The Laplace/Lidstone alpha, the cosine/kNN settings,regularisation) and chose the model
    - Test : Touched once,after all choices are frozen to report the final metrics
    
4. Checks to run and report
    - No group_id should appear in more than one split
    - Class proportions per split,compared with the full set (make a short table)
    - cross-split near duplicates and audit: For each test we document and compute the maximum TF-IDF cosine similarity to any train document. If many exceed 0.80, then the grouping missed something
    - Size and group_size distribution per split