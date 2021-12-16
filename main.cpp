#include <cmath>
#include <string>
#include <iostream> 
#include <benchmark/benchmark.h>

#include "dual.hpp"
#include "tensor.hpp"

#include "j2_material.hpp"
#include "neohookean_material.hpp"
#include "isotropic_linear_elastic_material.hpp"

std::string get_material_name(IsotropicLinearElasticMaterial) {
  return "isotropic linear elastic material";
}

std::string get_material_name(J2Material) {
  return "J2 material";
}

std::string get_material_name(NeoHookeanMaterial) {
  return "neohookean hyperelastic material";
}

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template<typename return_type, typename... Args>
return_type __enzyme_autodiff(Args...);

template<typename return_type, typename... Args>
return_type __enzyme_fwddiff(Args...);

template < typename material_type > 
void stress_evaluation_free_function(material_type material, const tensor< double, 3, 3 > & du_dx, tensor< double, 3, 3 > & stress) {
  stress = material.stress(du_dx);
}

template < typename material_type > 
void stress_evaluation_with_state_free_function(material_type material, typename material_type::State & state, const tensor< double, 3, 3 > & du_dx, tensor< double, 3, 3 > & stress) {
  stress = material.stress(du_dx, state);
}

template < typename material_type > 
void check_derivatives(material_type material) {

  std::cout << "verifying correctness of derivatives for: " << get_material_name(material) << " ... " << std::endl;

  double epsilon = 1.0e-8;
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};
  tensor<double, 3, 3> perturbation = {{{1.0, 0.2, 0.8}, {2.0, 0.1, 0.3}, {0.4, 0.2, 0.7}}};
  tensor< dual< double >, 3, 3 > du_dx_and_perturbation;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      du_dx_and_perturbation[i][j].value = du_dx[i][j];
      du_dx_and_perturbation[i][j].gradient = perturbation[i][j];
    }
  }

  auto dsigma1 = (material.stress(du_dx + epsilon * perturbation) - material.stress(du_dx - epsilon * perturbation)) / (2.0 * epsilon);
  auto dsigma2 = ddot(material.gradient(du_dx), perturbation);
  auto dsigma3 = material.action_of_gradient(du_dx, perturbation);
  auto dsigma4 = get_gradient(material.stress_AD(du_dx_and_perturbation));

  if (norm(dsigma1 - dsigma2) < (1.0e-5 * norm(dsigma1))) {
    std::cout << "- action of gradient evaluation by double contraction consistent with finite difference approximation" << std::endl;
  } else {
    std::cout << "- action of gradient evaluation by double contraction NOT consistent with finite difference approximation" << std::endl;
    exit(1);
  }

  if (norm(dsigma1 - dsigma3) < (1.0e-5 * norm(dsigma1))) {
    std::cout << "- direct action of gradient evaluation consistent with finite difference approximation" << std::endl;
  } else {
    std::cout << "- direct action of gradient evaluation NOT consistent with finite difference approximation" << std::endl;
    exit(1);
  }

  if (norm(dsigma1 - dsigma4) < (1.0e-5 * norm(dsigma1))) {
    std::cout << "- action of gradient evaluation by dual number class consistent with finite difference approximation" << std::endl;
  } else {
    std::cout << "- action of gradient evaluation by dual number class NOT consistent with finite difference approximation" << std::endl;
    exit(1);
  }

  tensor<double, 3, 3, 3, 3> gradient1{};

  tensor<double, 3, 3> unused{};
  tensor<double, 3, 3> dir{};

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      dir[i][j] = 1;
      __enzyme_autodiff< void >(stress_evaluation_free_function<material_type>,
          enzyme_const, material,
          enzyme_dup, &du_dx, &gradient1[i][j],
          enzyme_dup, &unused, &dir);
      dir[i][j] = 0;
    }
  }

  tensor<double, 3, 3, 3, 3> gradient2 = material.gradient(du_dx);
  tensor<double, 3, 3, 3, 3> gradient3 = get_gradient(material.stress_AD(make_dual(du_dx)));

  if (norm(gradient1 - gradient2) < (1.0e-5 * norm(gradient1))) {
    std::cout << "- gradient tensor by implementation of symbolic expression consistent with enzyme" << std::endl;
  } else {
    std::cout << "- gradient tensor by implementation of symbolic expression NOT consistent with enzyme" << std::endl;
    exit(1);
  }

  if (norm(gradient1 - gradient3) < (1.0e-5 * norm(gradient1))) {
    std::cout << "- gradient tensor by dual numbers consistent with enzyme" << std::endl;
  } else {
    std::cout << "- gradient tensor by dual numbers consistent with enzyme" << std::endl;
    exit(1);
  }

  std::cout << "all tests for this material PASS" << std::endl;
  std::cout << std::endl;

}

J2Material::State copy(J2Material::State state) { return state; } 

void check_derivatives(J2Material material) {

  std::cout << "verifying correctness of derivatives for: " << get_material_name(material) << " ... " << std::endl;

  double epsilon = 1.0e-8;
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};
  tensor<double, 3, 3> perturbation = {{{1.0, 0.2, 0.8}, {2.0, 0.1, 0.3}, {0.4, 0.2, 0.7}}};
  tensor< dual< double >, 3, 3 > du_dx_and_perturbation;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      du_dx_and_perturbation[i][j].value = du_dx[i][j];
      du_dx_and_perturbation[i][j].gradient = perturbation[i][j];
    }
  }

  J2Material::State state_before{ 
    tensor< double, 3, 3 >{}, // back stress
    tensor< double, 3, 3 >{}, // elastic strain
    0.0,                      // plastic strain
    0.0,                      // plastic strain increment
    0.0                       // trial J2 stress
  };

  J2Material::State discard[] = {copy(state_before), copy(state_before), copy(state_before)};

  J2Material::State state_after = copy(state_before);

  material.stress(du_dx, state_after);

  auto dsigma1 = (material.stress(du_dx + epsilon * perturbation, discard[0]) - 
                  material.stress(du_dx - epsilon * perturbation, discard[1])) / (2.0 * epsilon);
  auto dsigma2 = ddot(material.gradient(state_after), perturbation);
  auto dsigma3 = get_gradient(material.stress_AD(du_dx_and_perturbation, discard[2]));

  if (norm(dsigma1 - dsigma2) < (1.0e-5 * norm(dsigma1))) {
    std::cout << "- action of gradient evaluation by double contraction consistent with finite difference approximation" << std::endl;
  } else {
    std::cout << "- action of gradient evaluation by double contraction NOT consistent with finite difference approximation" << std::endl;
    exit(1);
  }

  if (norm(dsigma1 - dsigma3) < (1.0e-5 * norm(dsigma1))) {
    std::cout << "- action of gradient evaluation by dual number class consistent with finite difference approximation" << std::endl;
  } else {
    std::cout << "- action of gradient evaluation by dual number class NOT consistent with finite difference approximation" << std::endl;
    exit(1);
  }

  tensor<double, 3, 3, 3, 3> gradient1{};

  tensor<double, 3, 3> unused{};
  tensor<double, 3, 3> dir{};

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      auto state = copy(state_before);
      dir[i][j] = 1;
      __enzyme_autodiff< void >(stress_evaluation_with_state_free_function<J2Material>,
          enzyme_const, material,
          enzyme_const, state,
          enzyme_dup, &du_dx, &gradient1[i][j],
          enzyme_dup, &unused, &dir);
      dir[i][j] = 0;
    }
  }

  tensor<double, 3, 3, 3, 3> gradient2 = material.gradient(state_after);

  auto state = copy(state_before);
  tensor<double, 3, 3, 3, 3> gradient3 = get_gradient(material.stress_AD(make_dual(du_dx), state));

  if (norm(gradient1 - gradient2) < (1.0e-5 * norm(gradient1))) {
    std::cout << "- gradient tensor by implementation of symbolic expression consistent with enzyme" << std::endl;
  } else {
    std::cout << "- gradient tensor by implementation of symbolic expression NOT consistent with enzyme" << std::endl;
    exit(1);
  }

  if (norm(gradient1 - gradient3) < (1.0e-5 * norm(gradient1))) {
    std::cout << "- gradient tensor by dual numbers consistent with enzyme" << std::endl;
  } else {
    std::cout << "- gradient tensor by dual numbers consistent with enzyme" << std::endl;
    exit(1);
  }

  std::cout << "all tests for this material PASS" << std::endl;
  std::cout << std::endl;

}

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

static void stress_calculation_direct(benchmark::State& state) {

  double epsilon = 1.0e-8;
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};

  double C1 = 100.0;
  double D1 = 50.0;
  NeoHookeanMaterial material{C1, D1};

  for (auto _ : state) {
    auto sigma = material.stress(du_dx);
    benchmark::DoNotOptimize(sigma);
    du_dx[0][0] += 0.001;
  }
}
BENCHMARK(stress_calculation_direct);

static void gradient_calculation_enzyme(benchmark::State& state) {

  double epsilon = 1.0e-8;
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};
  tensor<double, 3, 3> perturbation = {{{1.0, 0.2, 0.8}, {2.0, 0.1, 0.3}, {0.4, 0.2, 0.7}}};

  double C1 = 100.0;
  double D1 = 50.0;

  tensor<double, 3, 3, 3, 3> gradient{};

  tensor<double, 3, 3> sigma{};
  tensor<double, 3, 3> dir{};

  for (auto _ : state) {

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
    du_dx[0][0] += sigma[0][0] * 0.001;
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
    du_dx[0][0] += 0.0001;
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

  for (auto _ : state) {
    auto gradient = stress_calculation_AD(make_dual(du_dx), C1, D1);
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

static void action_of_gradient_calculation_symbolic(benchmark::State& state) {

  double epsilon = 1.0e-8;
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};
  tensor<double, 3, 3> perturbation = {{{1.0, 0.2, 0.8}, {2.0, 0.1, 0.3}, {0.4, 0.2, 0.7}}};

  double C1 = 100.0;
  double D1 = 50.0;

  NeoHookeanMaterial material{C1, D1};

  for (auto _ : state) {
    auto dsigma = material.action_of_gradient(du_dx, perturbation);
    benchmark::DoNotOptimize(dsigma);
  }
}
BENCHMARK(action_of_gradient_calculation_symbolic);

static void action_of_gradient_calculation_precomputed_gradient(benchmark::State& state) {

  double epsilon = 1.0e-8;
  tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};
  tensor<double, 3, 3> perturbation = {{{1.0, 0.2, 0.8}, {2.0, 0.1, 0.3}, {0.4, 0.2, 0.7}}};

  double C1 = 100.0;
  double D1 = 50.0;

  NeoHookeanMaterial material{C1, D1};

  auto gradient = material.gradient(du_dx);

  for (auto _ : state) {
    auto dsigma = ddot(gradient, perturbation);
    benchmark::DoNotOptimize(dsigma);
  }
}
BENCHMARK(action_of_gradient_calculation_precomputed_gradient);

int main(int argc, char * argv[]) {

  check_derivatives(IsotropicLinearElasticMaterial{100.0, 50.0});
  check_derivatives(NeoHookeanMaterial{100.0, 50.0});
  check_derivatives(J2Material{1000.0, 0.25, 10, 10, 0.1});

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

}
