#pragma once

#include <iostream>
#include <type_traits>

#include "for_constexpr.hpp"

#include "dual.hpp"

#define HOST_DEVICE
#define SUPPRESS_NVCC_HOSTDEVICE_WARNING

// clang-format off
namespace detail {
template <typename T, typename i0_t>
HOST_DEVICE constexpr auto get(const T& values, i0_t i0) { return values[i0]; }

template <typename T, typename i0_t, typename i1_t>
HOST_DEVICE constexpr auto get(const T& values, i0_t i0, i1_t i1) { return values[i0][i1]; }

template <typename T, typename i0_t, typename i1_t, typename i2_t>
HOST_DEVICE constexpr auto get(const T& values, i0_t i0, i1_t i1, i2_t i2) { return values[i0][i1][i2]; }

template <typename T, typename i0_t, typename i1_t, typename i2_t, typename i3_t>
HOST_DEVICE constexpr auto get(const T& values, i0_t i0, i1_t i1, i2_t i2, i3_t i3) { return values[i0][i1][i2][i3]; }

template <typename T, typename i0_t, typename i1_t, typename i2_t, typename i3_t, typename i4_t >
HOST_DEVICE constexpr auto get(const T& values, i0_t i0, i1_t i1, i2_t i2, i3_t i3, i4_t i4) { return values[i0][i1][i2][i3][i4]; }

template <typename T, typename i0_t>
HOST_DEVICE constexpr auto& get(T& values, i0_t i0) { return values[i0]; }

template <typename T, typename i0_t, typename i1_t>
HOST_DEVICE constexpr auto& get(T& values, i0_t i0, i1_t i1) { return values[i0][i1]; }

template <typename T, typename i0_t, typename i1_t, typename i2_t>
HOST_DEVICE constexpr auto& get(T& values, i0_t i0, i1_t i1, i2_t i2) { return values[i0][i1][i2]; }

template <typename T, typename i0_t, typename i1_t, typename i2_t, typename i3_t>
HOST_DEVICE constexpr auto& get(T& values, i0_t i0, i1_t i1, i2_t i2, i3_t i3) { return values[i0][i1][i2][i3]; }

template <typename T, typename i0_t, typename i1_t, typename i2_t, typename i3_t, typename i4_t >
HOST_DEVICE constexpr auto& get(T& values, i0_t i0, i1_t i1, i2_t i2, i3_t i3, i4_t i4) { return values[i0][i1][i2][i3][i4]; }

template <int n>
using always_int = int;

}

template <typename T, int... n>
struct tensor;

template <typename T, int n>
struct tensor<T, n> {
  using type = T;
  static constexpr int ndim = 1;
  static constexpr int first_dim = n;

  HOST_DEVICE constexpr auto& operator[](int i) { return value[i]; };
  HOST_DEVICE constexpr auto operator[](int i) const { return value[i]; };

  template <typename S>
  HOST_DEVICE constexpr auto& operator()(S i) { return detail::get(value, i); }

  template <typename S>
  HOST_DEVICE constexpr auto operator()(S i) const { return detail::get(value, i); }

  T value[n];
};

template <typename T, int first, int... rest>
struct tensor<T, first, rest...> {
  using type = T;
  static constexpr int ndim = 1 + sizeof...(rest);
  static constexpr int first_dim = first;

  HOST_DEVICE constexpr auto& operator[](int i) { return value[i]; };
  HOST_DEVICE constexpr auto operator[](int i) const { return value[i]; };

  template <typename... S>
  HOST_DEVICE constexpr auto& operator()(S... i) { return detail::get(value, i...); };

  template <typename... S>
  HOST_DEVICE constexpr auto operator()(S... i) const { return detail::get(value, i...); };

  tensor<T, rest...> value[first];
};

template <typename T, int n1>
tensor(const T (&data)[n1]) -> tensor<T, n1>;

template <typename T, int n1, int n2>
tensor(const T (&data)[n1][n2]) -> tensor<T, n1, n2>;

struct zero {
  HOST_DEVICE operator double() { return 0.0; }

  template <typename T, int... n>
  HOST_DEVICE operator tensor<T, n...>() { return tensor<T, n...>{}; }

  template <typename... T>
  HOST_DEVICE auto operator()(T...) { return zero{}; }

  template <typename T>
  HOST_DEVICE auto operator=(T) { return zero{}; }
};

HOST_DEVICE constexpr auto operator+(zero, zero) { return zero{}; }

template <typename T>
HOST_DEVICE constexpr auto operator+(zero, T other) { return other; }

template <typename T>
HOST_DEVICE constexpr auto operator+(T other, zero) { return other; }

/////////////////////////////////////////////////

HOST_DEVICE constexpr auto operator-(zero) { return zero{}; }
HOST_DEVICE constexpr auto operator-(zero, zero) { return zero{}; }

template <typename T>
HOST_DEVICE constexpr auto operator-(zero, T other) { return -other; }

template <typename T>
HOST_DEVICE constexpr auto operator-(T other, zero) { return other; }

/////////////////////////////////////////////////

HOST_DEVICE constexpr auto operator*(zero, zero) { return zero{}; }

template <typename T>
HOST_DEVICE constexpr auto operator*(zero, T /*other*/) { return zero{}; }

template <typename T>
HOST_DEVICE constexpr auto operator*(T /*other*/, zero) { return zero{}; }

template <int i>
zero& get(zero& x) { return x; }

SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <typename lambda_type>
HOST_DEVICE constexpr auto make_tensor(lambda_type f) {
  using T = decltype(f());
  return tensor<T>{f()};
}

SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, typename lambda_type>
HOST_DEVICE constexpr auto make_tensor(lambda_type f) {
  using T = decltype(f(n1));
  tensor<T, n1> A{};
  for (int i = 0; i < n1; i++) { A(i) = f(i); }
  return A;
}

SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, int n2, typename lambda_type>
HOST_DEVICE constexpr auto make_tensor(lambda_type f) {
  using T = decltype(f(n1, n2));
  tensor<T, n1, n2> A{};
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      A(i, j) = f(i, j);
    }
  }
  return A;
}

SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, int n2, int n3, typename lambda_type>
HOST_DEVICE constexpr auto make_tensor(lambda_type f) {
  using T = decltype(f(n1, n2, n3));
  tensor<T, n1, n2, n3> A{};
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
        A(i, j, k) = f(i, j, k);
      }
    }
  }
  return A;
}

SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, int n2, int n3, int n4, typename lambda_type>
HOST_DEVICE constexpr auto make_tensor(lambda_type f) {
  using T = decltype(f(n1, n2, n3, n4));
  tensor<T, n1, n2, n3, n4> A{};
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
        for (int l = 0; l < n4; l++) {
          A(i, j, k, l) = f(i, j, k, l);
        }
      }
    }
  }
  return A;
}

SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, int n2, int n3, int n4, int n5, typename lambda_type>
HOST_DEVICE constexpr auto make_tensor(lambda_type f) {
  using T = decltype(f(n1, n2, n3, n4, n5));
  tensor<T, n1, n2, n3, n4, n5> A{};
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
        for (int l = 0; l < n4; l++) {
          for (int m = 0; m < n5; m++) {
            A(i, j, k, l, m) = f(i, j, k, l, m);
          }
        }
      }
    }
  }
  return A;
}

template <typename S, typename T, int... n>
HOST_DEVICE constexpr auto operator+(const tensor<S, n...>& A,
                                     const tensor<T, n...>& B) {
  tensor<decltype(S{} + T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) { C[i] = A[i] + B[i]; }
  return C;
}

template <typename T, int... n>
HOST_DEVICE constexpr auto operator-(const tensor<T, n...>& A) {
  tensor<T, n...> B{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) { B[i] = -A[i]; }
  return B;
}

template <typename S, typename T, int... n>
HOST_DEVICE constexpr auto operator-(const tensor<S, n...>& A,
                                     const tensor<T, n...>& B) {
  tensor<decltype(S{} + T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) { C[i] = A[i] - B[i]; }
  return C;
}

template <typename S, typename T, int... n,
          typename = std::enable_if_t< std::is_arithmetic_v<S> || is_dual_number<S>::value > >
HOST_DEVICE constexpr auto operator*(S scale, const tensor<T, n...>& A) {
  tensor<decltype(S{} * T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) { C[i] = scale * A[i]; }
  return C;
}

template <typename S, typename T, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value > >
HOST_DEVICE constexpr auto operator*(const tensor<T, n...>& A, S scale) {
  tensor<decltype(T{} * S{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) { C[i] = A[i] * scale; }
  return C;
}

template <typename S, typename T, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> > >
constexpr auto operator/(S scale, const tensor<T, n...>& A) {
  tensor<decltype(S{} * T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) { C[i] = scale / A[i]; }
  return C;
}

template <typename S, typename T, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> > >
constexpr auto operator/(const tensor<T, n...>& A, S scale) {
  tensor<decltype(T{} * S{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) { C[i] = A[i] / scale; }
  return C;
}

template <typename S, typename T, int... n>
constexpr auto& operator+=(tensor<S, n...>& A, const tensor<T, n...>& B) {
  for (int i = 0; i < tensor<S, n...>::first_dim; i++) { A[i] += B[i]; }
  return A;
}

template <typename T>
constexpr auto& operator+=(tensor<T>& A, const T& B) { return A.value += B; }

template <typename T>
constexpr auto& operator+=(tensor<T, 1>& A, const T& B) { return A.value += B; }

template <typename T>
constexpr auto& operator+=(tensor<T, 1, 1>& A, const T& B) { return A.value += B; }

template <typename T, int... n>
constexpr auto& operator+=(tensor<T, n...>& A, zero) { return A; }

template <typename S, typename T, int... n>
constexpr auto& operator-=(tensor<S, n...>& A, const tensor<T, n...>& B) {
  for (int i = 0; i < tensor<S, n...>::first_dim; i++) { A[i] -= B[i]; }
  return A;
}

template <typename T, int... n>
constexpr auto& operator-=(tensor<T, n...>& A, zero) { return A; }

template <typename T, int n>
constexpr auto outer(zero, const tensor<T, n>&) { return zero{}; }

template <typename T, int n>
constexpr auto outer(const tensor<T, n>&, zero) { return zero{}; }

template <typename S, typename T, int m, int n>
constexpr auto inner(const tensor<S, m, n>& A, const tensor<T, m, n>& B) {
  decltype(S{} * T{}) sum{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      sum += A[i][j] * B[i][j];
    }
  }
  return sum;
}

template <typename S, typename T, int m, int n, int p>
constexpr auto dot(const tensor<S, m, n>& A, const tensor<T, n, p>& B) {
  tensor<decltype(S{} * T{}), m, p> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < n; k++) {
        AB[i][j] = AB[i][j] + A[i][k] * B[k][j];
      }
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n>
constexpr auto dot(const tensor<S, m>& A, const tensor<T, m, n>& B) {
  tensor<decltype(S{} * T{}), n> AB{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      AB[i] = AB[i] + A[j] * B[j][i];
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n>
constexpr auto dot(const tensor<S, m, n>& A, const tensor<T, n>& B) {
  tensor<decltype(S{} * T{}), m> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i] = AB[i] + A[i][j] * B[j];
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n, int p>
constexpr auto dot(const tensor<S, m, n, p>& A, const tensor<T, p>& B) {
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        AB[i][j] += A[i][j][k] * B[k];
      }
    }
  }
  return AB;
}

template <typename S, typename T, int m, int... n>
constexpr auto dot(const tensor<S, m>& A, const tensor<T, m, n...>& B) {
  // this dot product function includes the vector * vector implementation and
  // the vector * tensor one, since clang emits an error about ambiguous
  // overloads if they are separate functions. The `if constexpr` expression
  // avoids using an `else` because that confuses nvcc (11.2) into thinking
  // there's not a return statement
  if constexpr (sizeof...(n) == 0) {
    decltype(S{} * T{}) AB{};
    for (int i = 0; i < m; i++) {
      AB += A[i] * B[i];
    }
    return AB;
  }

  if constexpr (sizeof...(n) > 0) {
    constexpr int dimensions[] = {n...};
    tensor<decltype(S{} * T{}), n...> AB{};
    for (int i = 0; i < dimensions[0]; i++) {
      for (int j = 0; j < m; j++) {
        AB[i] = AB[i] + A[j] * B[j][i];
      }
    }
    return AB;
  }
}

template <typename S, typename T, typename U, int m, int n>
constexpr auto dot(const tensor<S, m>& u, const tensor<T, m, n>& A,
                   const tensor<U, n>& v) {
  decltype(S{} * T{} * U{}) uAv{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      uAv += u[i] * A[i][j] * v[j];
    }
  }
  return uAv;
}

template <typename S, typename T, int m, int n, int p, int q>
constexpr auto ddot(const tensor<S, m, n, p, q>& A, const tensor<T, p, q>& B) {
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        for (int l = 0; l < q; l++) {
          AB[i][j] += A[i][j][k][l] * B[k][l];
        }
      }
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n, int p>
constexpr auto ddot(const tensor<S, m, n, p>& A, const tensor<T, n, p>& B) {
  tensor<decltype(S{} * T{}), m> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        AB[i] += A[i][j][k] * B[j][k];
      }
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n>
constexpr auto ddot(const tensor<S, m, n>& A, const tensor<T, m, n>& B) {
  decltype(S{} * T{}) AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB += A[i][j] * B[i][j];
    }
  }
  return AB;
}

template <typename S, typename T, int... m, int... n>
constexpr auto operator*(const tensor<S, m...>& A, const tensor<T, n...>& B) { return dot(A, B); }

template <typename S, typename T, int m, int n >
constexpr auto elementwise_multiplication(tensor<S, m, n>& A, const tensor<T, m, n>& B) { 
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] *= B[i][j];
    }
  }
}

template <typename S, typename T, int m, int n, int p >
constexpr auto elementwise_multiplication(tensor<S, m, n, p>& A, const tensor<T, m, n, p>& B) { 
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        A[i][j][k] *= B[i][j][k];
      }
    }
  }
}

template <typename T, int m>
constexpr auto sqnorm(const tensor<T, m>& A) {
  T total{};
  for (int i = 0; i < m; i++) { total += A[i] * A[i]; }
  return total;
}

template <typename T, int m, int n>
constexpr auto sqnorm(const tensor<T, m, n>& A) {
  T total{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      total += A[i][j] * A[i][j];
    }
  }
  return total;
}

template <typename T, int... n>
auto norm(const tensor<T, n...>& A) { return sqrt(sqnorm(A)); }

template <typename T, int... n>
auto normalize(const tensor<T, n...>& A) { return A / norm(A); }

template <typename T, int n>
constexpr auto tr(const tensor<T, n, n>& A) {
  T trA{};
  for (int i = 0; i < n; i++) {
    trA = trA + A[i][i];
  }
  return trA;
}

template <typename T, int n>
constexpr auto sym(const tensor<T, n, n>& A) {
  tensor<T, n, n> symA{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      symA[i][j] = 0.5 * (A[i][j] + A[j][i]);
    }
  }
  return symA;
}

template <typename T, int n>
constexpr auto dev(const tensor<T, n, n>& A) {
  auto devA = A;
  auto trA = tr(A);
  for (int i = 0; i < n; i++) {
    devA[i][i] -= trA / n;
  }
  return devA;
}

template <int dim>
HOST_DEVICE constexpr tensor<double, dim, dim> Identity() {
  tensor<double, dim, dim> I{};
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      I[i][j] = (i == j);
    }
  }
  return I;
}

template <typename T, int m, int n>
constexpr auto transpose(const tensor<T, m, n>& A) {
  tensor<T, n, m> AT{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      AT[i][j] = A[j][i];
    }
  }
  return AT;
}

template <typename T>
constexpr auto det(const tensor<T, 2, 2>& A) {
  return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

template <typename T>
constexpr auto det(const tensor<T, 3, 3>& A) {
  return A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
         A[0][2] * A[1][0] * A[2][1] - A[0][0] * A[1][2] * A[2][1] -
         A[0][1] * A[1][0] * A[2][2] - A[0][2] * A[1][1] * A[2][0];
}

namespace detail {
  template < typename T >
  auto abs(T x) { return (x < 0) ? -x : x; }

  template < typename T1, typename T2 >
  void swap(T1 & x1, T2 & x2) { auto tmp = x1; x1 = x2; x2 = tmp; }
}

template <typename T, int n>
constexpr tensor<T, n> linear_solve(tensor<T, n, n> A, const tensor<T, n> b) {

  tensor<T, n> x{};

  for (int i = 0; i < n; i++) {
    // Search for maximum in this column
    double max_val = detail::abs(A[i][i]);

    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      if (detail::abs(A[j][i]) > max_val) {
        max_val = detail::abs(A[j][i]);
        max_row = j;
      }
    }

    detail::swap(b[max_row], b[i]);
    detail::swap(A[max_row], A[i]);

    // zero entries below in this column
    for (int j = i + 1; j < n; j++) {
      double c = -A[j][i] / A[i][i];
      A[j] += c * A[i];
      b[j] += c * b[i];
      A[j][i] = 0;
    }
  }

  // Solve equation Ax=b for an upper triangular matrix A
  for (int i = n - 1; i >= 0; i--) {
    x[i] = b[i] / A[i][i];
    for (int j = i - 1; j >= 0; j--) {
      b[j] -= A[j][i] * x[i];
    }
  }

  return x;
}

template <typename T>
constexpr tensor<T, 2, 2> inv(const tensor<T, 2, 2>& A) {
  T inv_detA(1.0 / det(A));

  tensor<T, 2, 2> invA{};

  invA[0][0] = A[1][1] * inv_detA;
  invA[0][1] = -A[0][1] * inv_detA;
  invA[1][0] = -A[1][0] * inv_detA;
  invA[1][1] = A[0][0] * inv_detA;

  return invA;
}

template < typename T >
constexpr tensor<T, 3, 3> inv(const tensor<T, 3, 3>& A) {
  auto inv_detA = 1.0 / det(A);

  tensor<T, 3, 3> invA{};

  invA[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_detA;
  invA[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_detA;
  invA[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_detA;
  invA[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_detA;
  invA[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_detA;
  invA[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_detA;
  invA[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_detA;
  invA[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_detA;
  invA[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_detA;

  return invA;
}


template <typename T, int n>
constexpr tensor<T, n, n> inv(const tensor<T, n, n>& A) {

  tensor<double, n, n> B = Identity<n>();

  for (int i = 0; i < n; i++) {
    // Search for maximum in this column
    double max_val = detail::abs(A[i][i]);

    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      if (detail::abs(A[j][i]) > max_val) {
        max_val = detail::abs(A[j][i]);
        max_row = j;
      }
    }

    detail::swap(B[max_row], B[i]);
    detail::swap(A[max_row], A[i]);

    // zero entries below in this column
    for (int j = i + 1; j < n; j++) {
      if (A[j][i] != 0.0) {
        double c = -A[j][i] / A[i][i];
        A[j] += c * A[i];
        B[j] += c * B[i];
        A[j][i] = 0;
      }
    }
  }

  // upper triangular solve
  for (int i = n - 1; i >= 0; i--) {
    B[i] = B[i] / A[i][i];
    for (int j = i - 1; j >= 0; j--) {
      if (A[j][i] != 0.0) {
        B[j] -= A[j][i] * B[i];
      }
    }
  }

  return B;
}


template <typename T, int... n>
auto& operator<<(std::ostream& out, const tensor<T, n...>& A) {
  out << '{' << A[0];
  for (int i = 1; i < tensor<T, n...>::first_dim; i++) {
    out << ", " << A[i];
  }
  out << '}';
  return out;
}

template <int n>
constexpr auto chop(const tensor<double, n>& A) {
  auto copy = A;
  for (int i = 0; i < n; i++) {
    if (copy[i] * copy[i] < 1.0e-20) {
      copy[i] = 0.0;
    }
  }
  return copy;
}

template <int m, int n>
constexpr auto chop(const tensor<double, m, n>& A) {
  auto copy = A;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (copy[i][j] * copy[i][j] < 1.0e-20) {
        copy[i][j] = 0.0;
      }
    }
  }
  return copy;
}

template <typename gradient_type, int n>
auto inv(tensor<dual<gradient_type>, n, n> A) {
  auto invA = inv(get_value(A));
  return make_tensor<n, n>([&](int i, int j) {
    auto value = invA[i][j];
    gradient_type gradient{};
    for (int k = 0; k < n; k++) {
      for (int l = 0; l < n; l++) {
        gradient -= invA[i][k] * A[k][l].gradient * invA[l][j];
      }
    }
    return dual<gradient_type>{value, gradient};
  });
}

template <int... n>
constexpr auto make_dual(const tensor<double, n...>& A) {
  tensor<dual<tensor<double, n...>>, n...> A_dual{};
  for_constexpr<n...>([&](auto... i) {
    A_dual(i...).value = A(i...);
    A_dual(i...).gradient(i...) = 1.0;
  });
  return A_dual;
}

/// @cond
namespace detail {

template <typename T1, typename T2>
struct outer_prod;

template <int... m, int... n>
struct outer_prod<tensor<double, m...>, tensor<double, n...>> {
  using type = tensor<double, m..., n...>;
};

template <int... n>
struct outer_prod<double, tensor<double, n...>> {
  using type = tensor<double, n...>;
};

template <int... n>
struct outer_prod<tensor<double, n...>, double> {
  using type = tensor<double, n...>;
};

template <>
struct outer_prod<double, double> {
  using type = tensor<double>;
};

template <typename T>
struct outer_prod<zero, T> {
  using type = zero;
};

template <typename T>
struct outer_prod<T, zero> {
  using type = zero;
};

}  // namespace detail
/// @endcond

/**
 * @brief a type function that returns the tensor type of an outer product of
 * two tensors
 * @tparam T1 the first argument to the outer product
 * @tparam T2 the second argument to the outer product
 */
template <typename T1, typename T2>
using outer_product_t = typename detail::outer_prod<T1, T2>::type;

/**
 * @brief Retrieves a value tensor from a tensor of dual numbers
 * @param[in] arg The tensor of dual numbers
 */
template <typename T, int... n>
HOST_DEVICE auto get_value(const tensor<dual<T>, n...>& arg) {
  tensor<double, n...> value{};
  for_constexpr<n...>([&](auto... i) { value(i...) = arg(i...).value; });
  return value;
}

/**
 * @brief Retrieves the gradient component of a double (which is nothing)
 * @return The sentinel, @see zero
 */
HOST_DEVICE auto get_gradient(double /* arg */) { return zero{}; }

/**
 * @brief Retrieves a gradient tensor from a tensor of dual numbers
 * @param[in] arg The tensor of dual numbers
 */
template <int... n>
HOST_DEVICE auto get_gradient(const tensor<dual<double>, n...>& arg) {
  tensor<double, n...> g{};
  for_constexpr<n...>([&](auto... i) { g(i...) = arg(i...).gradient; });
  return g;
}

/// @overload
template <int... n, int... m>
HOST_DEVICE auto get_gradient(
    const tensor<dual<tensor<double, m...>>, n...>& arg) {
  tensor<double, n..., m...> g{};
  for_constexpr<n...>([&](auto... i) { g(i...) = arg(i...).gradient; });
  return g;
}

/**
 * @brief evaluate the change (to first order) in a function, f, given a small
 * change in the input argument, dx.
 */
HOST_DEVICE constexpr auto chain_rule(const zero /* df_dx */,
                                      const zero /* dx */) {
  return zero{};
}

/**
 * @overload
 * @note this overload implements a no-op for the case where the gradient w.r.t.
 * an input argument is identically zero
 */
template <typename T>
HOST_DEVICE constexpr auto chain_rule(const zero /* df_dx */,
                                      const T /* dx */) {
  return zero{};
}

/**
 * @overload
 * @note this overload implements a no-op for the case where the small change is
 * indentically zero
 */
template <typename T>
HOST_DEVICE constexpr auto chain_rule(const T /* df_dx */,
                                      const zero /* dx */) {
  return zero{};
}

/**
 * @overload
 * @note for a scalar-valued function of a scalar, the chain rule is just
 * multiplication
 */
HOST_DEVICE constexpr auto chain_rule(const double df_dx, const double dx) {
  return df_dx * dx;
}

/**
 * @overload
 * @note for a tensor-valued function of a scalar, the chain rule is just scalar
 * multiplication
 */
template <int... n>
HOST_DEVICE constexpr auto chain_rule(const tensor<double, n...>& df_dx,
                                      const double dx) {
  return df_dx * dx;
}

/**
 * @overload
 * @note for a scalar-valued function of a tensor, the chain rule is the inner
 * product
 */
template <int... n>
HOST_DEVICE constexpr auto chain_rule(const tensor<double, n...>& df_dx,
                                      const tensor<double, n...>& dx) {
  double total{};
  for_constexpr<n...>([&](auto... i) { total += df_dx(i...) * dx(i...); });
  return total;
}

/**
 * @overload
 * @note for a vector-valued function of a tensor, the chain rule contracts over
 * all indices of dx
 */
template <int m, int... n>
HOST_DEVICE constexpr auto chain_rule(const tensor<double, m, n...>& df_dx,
                                      const tensor<double, n...>& dx) {
  tensor<double, m> total{};
  for (int i = 0; i < m; i++) {
    total[i] = chain_rule(df_dx[i], dx);
  }
  return total;
}

/**
 * @overload
 * @note for a matrix-valued function of a tensor, the chain rule contracts over
 * all indices of dx
 */
template <int m, int n, int... p>
HOST_DEVICE auto chain_rule(const tensor<double, m, n, p...>& df_dx,
                            const tensor<double, p...>& dx) {
  tensor<double, m, n> total{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      total[i][j] = chain_rule(df_dx[i][j], dx);
    }
  }
  return total;
}
// clang-format on
