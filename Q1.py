def seed_test(SEED, X, y):
    from sklearn.model_selection import train_test_split

    for i in range(5):
        X_train_1, X_test_1, y_train_2, y_test_1 = train_test_split(X, y, test_size = 0.3,random_state = SEED) 
        print(y_test_1)

    print("*"*15)

    for i in range(5): 
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size = 0.3,random_state = SEED)
        print(y_test_2)

print("seed 10")
X = range(10)
y = range(10)
SEED = 10
seed_test(SEED, X, y)

print("*"*15)
print("seed 1")
X = range(10)
y = range(10)
SEED = 1
print(seed_test(SEED, X, y))
