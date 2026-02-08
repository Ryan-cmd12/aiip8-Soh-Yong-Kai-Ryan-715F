from sklearn.impute import KNNImputer, SimpleImputer

DB_path = "data/phishing.db"
Table = "phishing.db"


Target = "label"
Numerical_cols = ['LineOfCode',
                'LargestLineLength',
                'NoOfURLRedirect',
                'NoOfSelfRedirect', 
                'NoOfPopup', 
                'NoOfiFrame', 
                'NoOfImage',
                'NoOfSelfRef',
                'NoOfExternalRef', 
                'Robots', 
                'IsResponsive',
                'DomainAgeMonths',
                ]
Categorical_cols = ['Industry',
                    'HostingProvider']

Numerical_Imputer_Strategy = KNNImputer(n_neighbors = 5, weights = "distance")
Categorical_Imputer_Strategy = SimpleImputer(strategy ="most_frequent")

Test_Size = 0.2
Random_State = 42

