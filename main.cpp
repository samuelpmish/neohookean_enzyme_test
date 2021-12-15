#include <cmath>
#include <iostream> 
#include <benchmark/benchmark.h>

#include "dual.hpp"
#include "tensor.hpp"

int enzyme_dup;
int enzyme_out;
int enzyme_const;

template<typename return_type, typename... Args>
return_type __enzyme_autodiff(Args...);

template<typename return_type, typename... Args>
return_type __enzyme_fwddiff(Args...);

void stress_calculation(const tensor<double, 3, 3> & du_dx, double C1, double D1, tensor< double, 3, 3 >& sigma) {
  static constexpr auto I = Identity<3>();
  double J = det(I + du_dx);
  double p = -2.0 * D1 * J * (J - 1);
  auto devB = dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
  sigma = -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
}

tensor<double, 3, 3> stress_calculation_return(const tensor<double, 3, 3> & du_dx, double C1, double D1) {
  static constexpr auto I = Identity<3>();
  double J = det(I + du_dx);
  double p = -2.0 * D1 * J * (J - 1);
  auto devB = dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
  return -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
}

template < typename T >
tensor<T, 3, 3> stress_calculation_AD(const tensor<T, 3, 3> & du_dx, double C1, double D1) {
  static constexpr auto I = Identity<3>();
  auto J = det(I + du_dx);
  auto p = -2.0 * D1 * J * (J - 1);
  auto devB = dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
  return -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
}

// a compressible Rivlin model transcribed from:
// https://en.wikipedia.org/wiki/Neo-Hookean_solid#Cauchy_stress_in_terms_of_deformation_tensors
//
// expressions for derivatives were derived by hand (tests below are to verify correctness)
struct NeoHookeanMaterial {
  static constexpr auto I = Identity<3>();

  tensor<double, 3, 3> stress(tensor<double, 3, 3> du_dx) {
    double J = det(I + du_dx);
    double p = -2.0 * D1 * J * (J - 1);
    auto devB = dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
    return -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
  }

  // d(stress)_{ij} / d(du_dx)_{kl}
  tensor<double, 3, 3, 3, 3> gradient(tensor<double, 3, 3> du_dx) {
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
  tensor<double, 3, 3> action_of_gradient(tensor<double, 3, 3> du_dx, tensor<double, 3, 3> ddu_dx) {
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

  double C1, D1;
};

static void stress_calculation_direct(benchmark::State& state) {

  double epsilon = 1.0e-8;
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};

  double C1 = 100.0;
  double D1 = 50.0;
  NeoHookeanMaterial material{C1, D1};

  for (auto _ : state) {
    auto sigma = material.stress(du_dx);
    benchmark::DoNotOptimize(sigma);
  }
}
BENCHMARK(stress_calculation_direct);

static void gradient_calculation_enzyme(benchmark::State& state) {

  double epsilon = 1.0e-8;
  tensor<double, 3, 3> sigma{};
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};
  tensor<double, 3, 3> perturbation = {{{1.0, 0.2, 0.8}, {2.0, 0.1, 0.3}, {0.4, 0.2, 0.7}}};

  double C1 = 100.0;
  double D1 = 50.0;

  for (auto _ : state) {
    tensor<double, 3, 3, 3, 3> gradient{};

    tensor<double, 3, 3> sigma{};
    tensor<double, 3, 3> dir{};

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        dir[i][j] = 1;
        __enzyme_autodiff< void >(stress_calculation,
            enzyme_dup, &du_dx, &gradient[i][j],
            enzyme_out, C1,
            enzyme_out, D1,
            enzyme_dup, &sigma, &dir);
        dir[i][j] = 0;
      }
    }
    benchmark::DoNotOptimize(gradient);
  }
}
BENCHMARK(gradient_calculation_enzyme);

static void gradient_calculation_symbolic(benchmark::State& state) {

  double epsilon = 1.0e-8;
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};
  tensor<double, 3, 3> perturbation = {{{1.0, 0.2, 0.8}, {2.0, 0.1, 0.3}, {0.4, 0.2, 0.7}}};

  double C1 = 100.0;
  double D1 = 50.0;
  NeoHookeanMaterial material{C1, D1};

  for (auto _ : state) {
    auto gradient = material.gradient(du_dx);
    benchmark::DoNotOptimize(gradient);
  }
}
BENCHMARK(gradient_calculation_symbolic);

static void gradient_calculation_dual(benchmark::State& state) {

  double epsilon = 1.0e-8;
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};
  tensor<double, 3, 3> perturbation = {{{1.0, 0.2, 0.8}, {2.0, 0.1, 0.3}, {0.4, 0.2, 0.7}}};

  double C1 = 100.0;
  double D1 = 50.0;

  auto arg = make_dual(du_dx);

  for (auto _ : state) {

    auto gradient = stress_calculation_AD(arg, C1, D1);
    benchmark::DoNotOptimize(gradient);
  }
}
BENCHMARK(gradient_calculation_dual);

static void action_of_gradient_calculation_dual(benchmark::State& state) {

  double epsilon = 1.0e-8;
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};
  tensor<double, 3, 3> perturbation = {{{1.0, 0.2, 0.8}, {2.0, 0.1, 0.3}, {0.4, 0.2, 0.7}}};

  double C1 = 100.0;
  double D1 = 50.0;

  auto arg = make_dual(du_dx);

  for (auto _ : state) {
    tensor< dual< double >, 3, 3 > du_dx_and_perturbation;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        du_dx_and_perturbation[i][j].value = du_dx[i][j];
        du_dx_and_perturbation[i][j].gradient = perturbation[i][j];
      }
    }

    auto dsigma = get_gradient(stress_calculation_AD(du_dx_and_perturbation, C1, D1));
    benchmark::DoNotOptimize(dsigma);
  }
}
BENCHMARK(action_of_gradient_calculation_dual);

int main(int argc, char * argv[]) {
  double epsilon = 1.0e-8;
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};
  tensor<double, 3, 3> perturbation = {{{1.0, 0.2, 0.8}, {2.0, 0.1, 0.3}, {0.4, 0.2, 0.7}}};

  double C1 = 100.0;
  double D1 = 50.0;
  NeoHookeanMaterial material{C1, D1};

  auto dsigma1 = (material.stress(du_dx + epsilon * perturbation) - material.stress(du_dx - epsilon * perturbation)) / (2.0 * epsilon);
  auto dsigma2 = ddot(material.gradient(du_dx), perturbation);
  auto dsigma3 = material.action_of_gradient(du_dx, perturbation);

  tensor< dual< double >, 3, 3 > du_dx_and_perturbation;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      du_dx_and_perturbation[i][j].value = du_dx[i][j];
      du_dx_and_perturbation[i][j].gradient = perturbation[i][j];
    }
  }

  auto dsigma4 = get_gradient(stress_calculation_AD(du_dx_and_perturbation, C1, D1));

  //std::cout << dsigma1 << std::endl;
  //std::cout << dsigma2 << std::endl;
  //std::cout << dsigma3 << std::endl;
  //std::cout << dsigma4 << std::endl;
  //std::cout << std::endl;

  tensor<double, 3, 3, 3, 3> gradient1{};

  tensor<double, 3, 3> sigma{};
  tensor<double, 3, 3> dir{};

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      dir[i][j] = 1;
      __enzyme_autodiff< void >(stress_calculation,
          enzyme_dup, &du_dx, &gradient1[i][j],
          enzyme_out, C1,
          enzyme_out, D1,
          enzyme_dup, &sigma, &dir);
      dir[i][j] = 0;
    }
  }

  tensor<double, 3, 3, 3, 3> gradient2 = material.gradient(du_dx);
  // std::cout << gradient1 - gradient2 << std::endl;

  auto output = stress_calculation_AD(make_dual(du_dx), C1, D1);
  // std::cout << gradient1 - get_gradient(output) << std::endl;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      dir[i][j] = 1;
      __enzyme_autodiff< void >(stress_calculation,
          enzyme_dup, &du_dx, &gradient1[i][j],
          enzyme_out, C1,
          enzyme_out, D1,
          enzyme_dup, &sigma, &dir);
      dir[i][j] = 0;
    }
  }

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

}
