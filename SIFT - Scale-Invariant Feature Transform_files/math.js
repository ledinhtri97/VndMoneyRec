// Copyright (c) 2016, Dr. Edmund Weitz. All rights reserved.

var twoPi = 2 * Math.PI;

function Matrix (width, height, fill) {
  this.width = width;
  this.height = height;
  this.length = width * height;
  this.data = new Float64Array(new ArrayBuffer(8 * this.length));
  if (fill !== undefined)
    this.data.fill(fill);
}

Matrix.prototype.aref = function (i, j) {
  return this.data[j * this.width + i];
};

Matrix.prototype.copy = function () {
  var newMatrix = new Matrix(this.width, this.height);
  for (var i = 0; i < this.length; i++)
    newMatrix.data[i] = this.data[i];
  return newMatrix;
}

function mod (a, m) {
  var res = a % m;
  if (res < 0)
    res += m;
  return res;
}

function mod2Pi (x) {
  if (x < 0)
    return x + twoPi;
  if (x >= twoPi)
    return x - twoPi;
  return x;
}

function convolveCircular (kernel, arr, n) {
  n = n || 1;
  var arrLen = arr.length;
  var result = new Array(arrLen).fill(0);
  var i = 0;
  for (var offset = -Math.floor((kernel.length - 1) / 2); offset <= Math.floor(kernel.length / 2); offset++) {
    var factor = kernel[i++];
    for (var j = 0; j < arrLen; j++)
      result[j] += factor * arr[mod(j + offset, arrLen)];
  }
  return n <= 1 ? result : convolveCircular(kernel, result, n - 1);
}

function convolveGaussian (sigma, matrix) {
  var kernel = GaussKernel(sigma);
  return convolveY(kernel, convolveX(kernel, matrix));
}

function convolveX (kernel, matrix) {
  var matrixSize = matrix.length;
  var matrixWidth = matrix.width;
  var result = new Matrix(matrixWidth, matrix.height, 0);
  var matrixData = matrix.data;
  var resultData = result.data;
  var i, j, k = 0;
  for (var offset = -Math.floor((kernel.length - 1) / 2); offset <= Math.floor(kernel.length / 2); offset++) {
    var factor = kernel[k++];
    j = 0;
    for (var y = 0; y < matrixSize; y += matrixWidth) {
      // get rid of i
      for (var x = 0; x < matrixWidth; x++) {
        i = x + offset;
        if (i < 0)
          i = -i;
        else if (i >= matrixWidth)
          i = matrixWidth + matrixWidth - i - 1;
        resultData[j++] += factor * matrixData[y + i];
      }
    }
  }
  return result;
}

function convolveY (kernel, matrix) {
  var matrixSize = matrix.length;
  var matrixWidth = matrix.width;
  var matrixHeight = matrix.height;
  var result = new Matrix(matrixWidth, matrixHeight, 0);
  var matrixData = matrix.data;
  var resultData = result.data;
  var i, j, k = 0;
  for (var offset = -Math.floor((kernel.length - 1) / 2); offset <= Math.floor(kernel.length / 2); offset++) {
    var factor = kernel[k++];
    j = 0;
    // get rid of i
    for (var y = 0; y < matrixHeight; y++) {
      // can be improved:
      i = y + offset;
      if (i < 0)
        i = -i;
      else if (i >= matrixHeight)
        i = matrixHeight + matrixHeight - i - 1;
      i *= matrixWidth;
      for (var x = 0; x < matrixWidth; x++) {
        resultData[j++] += factor * matrixData[i + x];
      }
    }
  }
  return result;
}

function GaussKernel (sigma) {
  var size = Math.floor(4 * sigma);
  var denom = -2 * sigma * sigma;
  var kernel = new Array(2 * size + 1);
  var i = 0, sum = 0;
  for (var k = -size; k <= size; k++) {
    kernel[i] = Math.exp(k * k / denom);
    sum += kernel[i];
    i++;
  }
  i = 0;
  for (k = -size; k <= size; k++) {
    kernel[i++] /= sum;
  }
  return kernel;
}

function computeMinMax (matrix) {
  var max = -1e50;
  var min = 1e50;
  var len = matrix.length;
  var data = matrix.data;
  for (var i = 0; i < len; i++) {
    if (data[i] > max)
      max = data[i];
    if (data[i] < min)
      min = data[i];
  }
  return [min, max];
}

function computeGlobalMinMax (arr) {
  var all = [];
  arr.forEach(function (element) {
    all.push(Array.isArray(element) ? computeGlobalMinMax(element) : computeMinMax(element))
  });
  return [
    Math.min.apply(null, all.map(function (pair) { return pair[0]; })),
    Math.max.apply(null, all.map(function (pair) { return pair[1]; }))
  ];
}

function normalize (fromMatrix, minMax, range) {
  minMax = minMax || computeMinMax(fromMatrix);
  range = range || [0, 255];
  var matrix = fromMatrix.copy();
  var min = minMax[0];
  var minMaxSpan = minMax[1] - min;
  var start = range[0];
  var rangeSpan = range[1] - start;
  var len = matrix.length;
  var matrixData = matrix.data;
  for (var i = 0; i < len; i++)
    matrixData[i] = (matrixData[i] - min) / minMaxSpan * rangeSpan + start;
  return matrix;
}

function normalizeNumber (x, minMax, range) {
  range = range || [0, 255];
  var min = minMax[0];
  var minMaxSpan = minMax[1] - min;
  var start = range[0];
  var rangeSpan = range[1] - start;
  return (x - min) / minMaxSpan * rangeSpan + start;
}

function bilinearInterpolation (matrix) {
  var oldWidth = matrix.width;
  var oldHeight = matrix.height;
  var matrixData = matrix.data;
  var newWidth = 2 * oldWidth;
  var newHeight = 2 * oldHeight;
  var result = new Matrix(newWidth, newHeight);
  var resultData = result.data;
  var i = 0;
  var j = 0;
  var mEven = true;
  var nEven = true;
  var x, y = 1;
  for (var n = 0; n < newHeight; n++) {
    x = 1;
    for (var m = 0; m < newWidth; m++) {
      if (mEven) {
        if (nEven || x == oldWidth)
          resultData[i] = matrixData[j];
        else
          resultData[i] = 0.5 * (matrixData[j] + matrixData[j + 1]);
      } else {
        if (nEven) {
          if (y < oldHeight)
            resultData[i] = 0.5 * (matrixData[j] + matrixData[j + oldWidth]);
          else
            resultData[i] = matrixData[j];
        } else {
          if (x < oldWidth) {
            if (y < oldHeight)
              resultData[i] = 0.25 * (matrixData[j] + matrixData[j + 1] +
                                      matrixData[j + oldWidth] + matrixData[j + oldWidth + 1]);
            else
              resultData[i] = 0.5 * (matrixData[j] + matrixData[j + 1]);
          } else {
            if (y < oldHeight)
              resultData[i] = 0.5 * (matrixData[j] + matrixData[j + oldWidth]);
            else
              resultData[i] = matrixData[j];
          }
        }
      }
      i++;
      mEven = !mEven;
      if (mEven) {
        j++;
        x++;
      }
    }
    nEven = !nEven;
    if (!nEven)
      j -= oldWidth;
    if (nEven)
      y++;
  }
  return result;
}

function subsample (matrix) {
  var oldWidth = matrix.width;
  var oldHeight = matrix.height;
  var matrixData = matrix.data;
  var newWidth = Math.floor(oldWidth / 2);
  var newHeight = Math.floor(oldHeight / 2);
  var result = new Matrix(newWidth, newHeight);
  var resultData = result.data;
  var i = 0, j = 0;
  for (var y = 0; y < newHeight; y++) {
    for (var x = 0; x < newWidth; x++) {
      resultData[i++] = matrixData[j];
      j += 2;
    }
    j += oldWidth;
  }
  return result;
}

// assumes same size
function matrixDifference (matrix1, matrix2) {
  var width = matrix1.width;
  var height = matrix1.height;
  var size = width * height;
  var result = new Matrix(width, height);
  var resultData = result.data;
  var matrix1Data = matrix1.data;
  var matrix2Data = matrix2.data;

  for (var i = 0; i < size; i++)
    resultData[i] = matrix1Data[i] - matrix2Data[i];
  return result;
}

function gradient (point) {
  var o = point.octave;
  var s = point.scale;
  var m = point.xIndex;
  var n = point.yIndex;

  return [
      .5 * (differenceOfGaussians[o][s + 1].aref(m, n) - differenceOfGaussians[o][s - 1].aref(m, n)),
      .5 * (differenceOfGaussians[o][s].aref(m + 1, n) - differenceOfGaussians[o][s].aref(m - 1, n)),
      .5 * (differenceOfGaussians[o][s].aref(m, n + 1) - differenceOfGaussians[o][s].aref(m, n - 1))
  ];
}

function Hessian3D (point) {
  var o = point.octave;
  var s = point.scale;
  var m = point.xIndex;
  var n = point.yIndex;

  var h12 = .25 * (differenceOfGaussians[o][s + 1].aref(m + 1, n) -
                   differenceOfGaussians[o][s + 1].aref(m - 1, n) -
                   differenceOfGaussians[o][s - 1].aref(m + 1, n) +
                   differenceOfGaussians[o][s - 1].aref(m - 1, n));
  var h13 = .25 * (differenceOfGaussians[o][s + 1].aref(m, n + 1) -
                   differenceOfGaussians[o][s + 1].aref(m, n - 1) -
                   differenceOfGaussians[o][s - 1].aref(m, n + 1) +
                   differenceOfGaussians[o][s - 1].aref(m, n - 1));
  var h23 = .25 * (differenceOfGaussians[o][s].aref(m + 1, n + 1) -
                   differenceOfGaussians[o][s].aref(m + 1, n - 1) -
                   differenceOfGaussians[o][s].aref(m - 1, n + 1) +
                   differenceOfGaussians[o][s].aref(m - 1, n - 1));

  var subtract = 2 * differenceOfGaussians[o][s].aref(m, n);

  return [
    [
      differenceOfGaussians[o][s + 1].aref(m, n) + differenceOfGaussians[o][s - 1].aref(m, n) - subtract,
      h12,
      h13
    ],
    [
      h12,
      differenceOfGaussians[o][s].aref(m + 1, n) + differenceOfGaussians[o][s].aref(m - 1, n) - subtract,
      h23
    ],
    [
      h13,
      h23,
      differenceOfGaussians[o][s].aref(m, n + 1) + differenceOfGaussians[o][s].aref(m, n - 1) - subtract
    ]
  ];
}

function quadraticInterpolation (point) {
  var invHessian = invMatrix(Hessian3D(point));
  if (!invHessian)
    return undefined;
  var grad = gradient(point);
  var alpha = scalarMult(-1, matrixVector(invHessian, grad));
  var omega = differenceOfGaussians[point.octave][point.scale].aref(point.xIndex, point.yIndex) -
        .5 * dotProduct(grad, matrixVector(invHessian, grad));
  return [alpha, omega];
}

// inverts a 3x3 matrix
function invMatrix (A) {
  var denom = A[0][0]*A[1][1]*A[2][2] - A[0][0]*A[1][2]*A[2][1] - A[0][1]*A[1][0]*A[2][2] +
        A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1] - A[0][2]*A[1][1]*A[2][0];
  if (denom == 0)
    return undefined;
  return [
    [
      (A[1][1]*A[2][2] - A[1][2]*A[2][1]) / denom,
      (A[0][2]*A[2][1] - A[0][1]*A[2][2]) / denom,
      (A[0][1]*A[1][2] - A[0][2]*A[1][1]) / denom
    ],
    [
      (A[1][2]*A[2][0] - A[1][0]*A[2][2]) / denom,
      (A[0][0]*A[2][2] - A[0][2]*A[2][0]) / denom,
      (A[0][2]*A[1][0] - A[0][0]*A[1][2]) / denom
    ],
    [
      (A[1][0]*A[2][1] - A[1][1]*A[2][0]) / denom,
      (A[0][0]*A[2][1] - A[0][1]*A[2][0]) / denom,
      (A[0][0]*A[1][1] - A[0][1]*A[1][0]) / denom]
  ];
}

// returns the product of a 3x3 matrix with a vector
function matrixVector (A, v) {
  return [
    A[0][0]*v[0] + A[0][1]*v[1] + A[0][2]*v[2],
    A[1][0]*v[0] + A[1][1]*v[1] + A[1][2]*v[2],
    A[2][0]*v[0] + A[2][1]*v[1] + A[2][2]*v[2]
  ];
}

function dotProduct (v1, v2) {
  return v1[0]*v2[0] * v1[1]*v2[1] * v1[2]*v2[2];
}

function scalarMult (scalar, v) {
  return [scalar * v[0], scalar * v[1], scalar * v[2]];
}

function Hessian2D (point) {
  var o = point.octave;
  var s = point.scale;
  var m = point.xIndex;
  var n = point.yIndex;

  var h12 = .25 * (differenceOfGaussians[o][s].aref(m + 1, n + 1) -
                   differenceOfGaussians[o][s].aref(m + 1, n - 1) -
                   differenceOfGaussians[o][s].aref(m - 1, n + 1) +
                   differenceOfGaussians[o][s].aref(m - 1, n - 1));

  var subtract = 2 * differenceOfGaussians[o][s].aref(m, n);

  return [
    [
      differenceOfGaussians[o][s].aref(m + 1, n) +
      differenceOfGaussians[o][s].aref(m - 1, n) - subtract,
      h12
    ],
    [
      h12,
      differenceOfGaussians[o][s].aref(m, n + 1) +
      differenceOfGaussians[o][s].aref(m, n - 1) - subtract
    ]
  ];  
}

// computes the determinant of the 2x2 matrix
function determinant (matrix) {
  return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
}

function trace (matrix) {
  return matrix[0][0] + matrix[1][1];
}

function atan2 (x, y) {
  var res = Math.atan2(x, y);
  return mod2Pi(res);
}

function normSquared (x, y) {
  return x * x + y * y;
}

function norm (x, y) {
  return Math.sqrt(normSquared(x, y));
}

function Cube (width, height, depth, fill) {
  this.width = width;
  this.height = height;
  this.height = depth;
  this.length = width * height * depth;
  this.data = new Float64Array(new ArrayBuffer(8 * this.length));
  if (fill !== undefined)
    this.data.fill(fill);
}

Cube.prototype.aref = function (i, j, k) {
  return this.data[k * (this.height + this.width) + j * this.width + i];
};

Cube.prototype.incf = function (i, j, k, val) {
  this.data[k * (this.height + this.width) + j * this.width + i] += val;
};

Cube.prototype.norm = function () {
  var result = 0;
  for (var i = 0; i < this.length; i++)
    result += this.data[i] * this.data[i];
  return Math.sqrt(result);
}

function formatNumber (x, n) {
  n = n || 2;
  var factor = Math.round(Math.pow(10, n));
  var sign = x < 0 ? "- " : "  ";
  x = Math.abs(x);
  var frac = "" + (Math.round(factor * x) % factor);
  while (frac.length < n)
    frac = "0" + frac;
  return sign + Math.floor(Math.round(factor * x) / factor) + "." + frac;
}
