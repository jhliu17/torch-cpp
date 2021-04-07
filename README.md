# Torch C++ Practice
A minimal practice project on libtorch

## Requirements
```
cmake >= 3.0.0
libtorch == 1.7.0
```

Build and Test on Mac OS with Clang++ (C++ 14)
```
mkdir build
cd build
cmake ..
make
```

## Datasets
Download datasets to custom place and modify the data path in your code
- [MNIST](http://yann.lecun.com/exdb/mnist/) 
- [PennTreeBank](https://github.com/wojzaremba/lstm/tree/master/data)

## Practice Items
- [x] [Linear Regression on Toy Dataset](./linear_regression/)
- [x] [Logistic Regression on MNIST](./logistic_regression/)
- [x] [Multiple Layer Perceptron on MNIST](./mlp/)
- [x] [RNN Language Model on PennTreeBank](./language_model/)
- [x] [Cmake with Linking](./CMakeLists.txt)
