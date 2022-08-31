import numpy as np
import tensorflow as tf
from tensorflow import constant
from tensorflow.python.ops.gen_math_ops import add
import pandas as pd

def Tensoren():
    #0 Diemansion Tensor
    d0 = tf.ones((1,))

    #1 Dimension Tensor
    d1 = tf.ones((2,))

    #2 Dimension Tensor
    d2 = tf.ones((2,2))

    #3 Dimension Tensor
    d3 = tf.ones((2,2,2))

    print("3D Tensor")
    print(d3.numpy())

    print("\n2d tensor")
    print(d2.numpy())

    print("\n1d Tensor")
    print(d1.numpy())

    print("\n0D Tensor")
    print(d1.numpy())


def constants():
    # 2x3 Konstante output: [[2 2 2]
    #                        [2 2 2]]
    a = constant(2, shape=[2,3])

    # 2x2 konstante output: [[1 2]
    #                        [3 4]]
    b = constant([1,2,3,4], shape=[2,2])

    print("First Tensor constant")
    print(a.numpy())
    print("\nSecond Tensor constant")
    print(b.numpy())

#print("Tensoren")
#Tensoren()
#print("\nKonstanten:")
#constnten()

def other_tensors():
    st1 = tf.fill([3,3],7)
    st2 = tf.zeros([5,5])

    print(st1.numpy())
    print(st2.numpy())

def Variables():
    a0 = tf.Variable([1,2,3,4,5,6], dtype=tf.float32)

    #a1 = tf.Variable([1,2,3,4,5,6], dtype=tf.int16)

def computer_tensors():
    a0 = tf.Variable([1,2,3,4,5,6], dtype=tf.float32)
    a = tf.constant(2, tf.float32)

    c0 = tf.multiply(a0, a)
    c1 = a0 * a
    print(c1.numpy())


def addiotion_of_tensors():
    #0 Dimension constant
    a0 = constant([1])
    a1 = constant([2])

    #1 Dimension
    c1 = constant([1,2])
    c2 = constant([3,4])

    #2 Dimensions
    b0 = constant([[1,2], [3,4]])
    b1 = constant([[5,6], [7,8]])


    d0 = add(a0, a1)
    d1 = add(c1, c2)
    d2 = add(b0, b1)


    print(d0.numpy())
    print(d1.numpy())
    print(d2.numpy())

#addiotion_of_tensors()

board = [1,2,0,1,0,2,2,1,0]

field = tf.constant([[board[0],board[1],board[2]],[board[3],board[4],board[5]],[board[6],board[7], board[8]]], shape=(3,3))


print(field.numpy()) #print numpy field
