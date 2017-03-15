import pickle
from helper import transform, show_image, draw_class_distribution

training_file = './traffic-signs-data/train.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

#display_images(X_train, y_train, 5, 5)
draw_class_distribution(y_train)
print(transform(X_train[1000]).shape)