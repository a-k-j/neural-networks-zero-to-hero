For working with tensors in Torch we need to know few things:

1. Two tensors are “broadcastable” if the following rules hold:
- When iterating over the dimension sizes, starting at the trailing dimension (the last one), the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
- When a dimension size is 1, that dimension will be "stretched" to match the size of the other tensor.

2. If two tensors x, y are “broadcastable”, the resulting tensor size is calculated as follows:
- If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.
- Then, for each dimension size, the resulting dimension size is the max of the sizes of x and y along that dimension.

3. P = P / P.sum(1, keepdim = True)
Here keepdim = True is really important because:
- By default keepdim is False and if we forget to set it to true, the sum is still calculated correctly but the returned Tensor has dimension of just 27 as the 1 is squeezed out due to the default behaviour of Torch.
- If we try to divide P by this we still get a result because both tensors are broadcastable as:
    - P is 27 * 27 and P.sum(1) is 27
    - Right aligning them and then checking for conditions for broadcastability reveals: \
        27 27 \
        __ 27 \
    Since 27 matches and then after that a missing dimension is not an issue since according to rules.
    - Torch will add 1 for that missing dimension.
- The code runs successfully but we do not get the expected result as the division does not produce a Probability Distribution due to incorrect division.

4. torch.tensor build dtype torch.int64 and torch.tensor build dtype torch.float32

5. Generator should be reinitialised everytime before using to make sure consistency across notebooks.