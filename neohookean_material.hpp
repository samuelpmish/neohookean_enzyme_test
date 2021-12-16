#pragma once

#include "tensor.hpp"

// a compressible Rivlin model transcribed from:
// https://en.wikipedia.org/wiki/Neo-Hookean_solid#Cauchy_stress_in_terms_of_deformation_tensors
//
// expressions for derivatives were derived by hand (tests below are to verify correctness)
struct NeoHookeanMaterial {

  static constexpr auto I = Identity<3>();

  tensor<double, 3, 3> stress(tensor<double, 3, 3> du_dx) const {
    double J = det(I + du_dx);
    double p = -2.0 * D1 * J * (J - 1);
    auto devB = dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
    return -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
  }

  // d(stress)_{ij} / d(du_dx)_{kl}
  tensor<double, 3, 3, 3, 3> gradient(tensor<double, 3, 3> du_dx) const {
    tensor<double, 3, 3> F = I + du_dx;
    tensor<double, 3, 3> invF = inv(F);
    tensor<double, 3, 3> devB = dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
    double J = det(F);
    double coef = (C1 / pow(J, 5.0 / 3.0));
    // clang-format off
    return make_tensor<3, 3, 3, 3>([&](auto i, auto j, auto k, auto l) {
      return 2.0 * (D1 * J * (i == j) - (5.0 / 3.0) * coef * devB[i][j]) * invF[l][k] +
             2.0 * coef * ((i == k) * F[j][l] + F[i][l] * (j == k) - (2.0 / 3.0) * ((i == j) * F[k][l]));
    });
    // clang-format on
  }

  // d(stress)_{ij} := (d(stress)_ij / d(du_dx)_{kl}) * d(du_dx)_{kl}
  tensor<double, 3, 3> action_of_gradient(tensor<double, 3, 3> du_dx, tensor<double, 3, 3> ddu_dx) const {
    tensor<double, 3, 3> F = I + du_dx;
    tensor<double, 3, 3> invFT = inv(transpose(F));
    tensor<double, 3, 3> devB = dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
    double J = det(F);
    double coef = (C1 / pow(J, 5.0 / 3.0));
    double a1 = ddot(invFT, ddu_dx);
    double a2 = ddot(F, ddu_dx);

    return (2.0 * D1 * J * a1 - (4.0 / 3.0) * coef * a2) * I -
           ((10.0 / 3.0) * coef * a1) * devB +
           (2 * coef) * (dot(ddu_dx, transpose(F)) + dot(F, transpose(ddu_dx)));
  }

  template < typename T >
  tensor<T, 3, 3> stress_AD(tensor<T, 3, 3> du_dx) const {
    auto J = det(I + du_dx);
    auto p = -2.0 * D1 * J * (J - 1);
    auto devB = dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
    return -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
  }

  double C1, D1;
};

