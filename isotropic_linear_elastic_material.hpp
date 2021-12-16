#pragma once

#include "tensor.hpp"

struct IsotropicLinearElasticMaterial {

  static constexpr auto I = Identity<3>();

  tensor<double, 3, 3> stress(tensor<double, 3, 3> du_dx) {
    return lambda * tr(du_dx) * I + mu * (du_dx + transpose(du_dx));
  }

  // d(stress)_{ij} / d(du_dx)_{kl}
  tensor<double, 3, 3, 3, 3> gradient(tensor<double, 3, 3> /* du_dx */) {
    return make_tensor<3, 3, 3, 3>([&](auto i, auto j, auto k, auto l) {
      return lambda * (i == j) * (k == l) + mu * ((i == l) * (j == k) + (i == k) * (j == l));
    });
  }

  // d(stress)_{ij} := (d(stress)_ij / d(du_dx)_{kl}) * d(du_dx)_{kl}
  tensor<double, 3, 3> action_of_gradient(tensor<double, 3, 3> /*du_dx*/, tensor<double, 3, 3> ddu_dx) {
    // note: this material is linear, 
    // so its action-of-gradient is
    // the same as its stress evaluation
    return stress(ddu_dx); 
  }

  template < typename T >
  tensor<T, 3, 3> stress_AD(tensor<T, 3, 3> du_dx) {
    return lambda * tr(du_dx) * I + mu * (du_dx + transpose(du_dx));
  }

  double lambda, mu;

};

