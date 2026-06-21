#include <array>
#include <cstddef>
#include <tuple>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

/// Discretizes the given continuous A and B matrices.
///
/// @tparam Scalar Scalar type.
/// @tparam States Number of states.
/// @tparam Inputs Number of inputs.
/// @param cont_A Continuous system matrix.
/// @param cont_B Continuous input matrix.
/// @param dt Discretization timestep in seconds.
/// @return Discrete system matrix and discrete input matrix.
template <typename Scalar, int States, int Inputs>
std::tuple<Eigen::Matrix<Scalar, States, States>,
           Eigen::Matrix<Scalar, States, Inputs>>
discretize_ab(const Eigen::Matrix<Scalar, States, States>& cont_A,
              const Eigen::Matrix<Scalar, States, Inputs>& cont_B, Scalar dt) {
  // M = [A  B]
  //     [0  0]
  Eigen::Matrix<Scalar, States + Inputs, States + Inputs> M;
  M.template block<States, States>(0, 0) = cont_A;
  M.template block<States, Inputs>(0, States) = cont_B;
  M.template block<Inputs, States + Inputs>(States, 0).setZero();

  // ϕ = eᴹᵀ = [A_d  B_d]
  //           [ 0    I ]
  Eigen::Matrix<Scalar, States + Inputs, States + Inputs> ϕ = (M * dt).exp();

  return {ϕ.template block<States, States>(0, 0),
          ϕ.template block<States, Inputs>(0, States)};
}

/// Discretizes the given continuous Q matrix.
///
/// @tparam Scalar Scalar type.
/// @tparam States Number of states.
/// @param cont_A Continuous system matrix.
/// @param cont_Q Continuous process noise covariance matrix.
/// @param dt Discretization timestep in seconds.
/// @return Discrete process noise covariance matrix.
template <typename Scalar, int States>
Eigen::Matrix<Scalar, States, States> discretize_q(
    const Eigen::Matrix<Scalar, States, States>& cont_A,
    const Eigen::Matrix<Scalar, States, States>& cont_Q, Scalar dt) {
  // M = [−A  Q ]
  //     [ 0  Aᵀ]
  Eigen::Matrix<Scalar, 2 * States, 2 * States> M;
  M.template block<States, States>(0, 0) = -cont_A;
  M.template block<States, States>(0, States) = cont_Q;
  M.template block<States, States>(States, 0).setZero();
  M.template block<States, States>(States, States) = cont_A.transpose();

  // ϕ = eᴹᵀ = […  A_d⁻¹Q_d]
  //           [0     A_dᵀ ]
  Eigen::Matrix<Scalar, 2 * States, 2 * States> ϕ = (M * dt).exp();

  // Q_d = A_d(A_d⁻¹Q_d)
  Eigen::Matrix<Scalar, States, States> disc_Q =
      ϕ.template block<States, States>(States, States).transpose() *
      ϕ.template block<States, States>(0, States);

  return (disc_Q + disc_Q.transpose()) / Scalar(2);
}

/// Creates a covariance matrix from the given vector for use with Kalman
/// filters.
///
/// Each element is squared and placed on the covariance matrix diagonal.
///
/// @tparam Scalar Scalar type.
/// @tparam N Number of elements.
/// @param std_devs An array. For a Q matrix, its elements are the standard
///     deviations of each state from how the model behaves. For an R matrix,
///     its elements are the standard deviations for each output measurement.
/// @return Process noise or measurement noise covariance matrix.
template <typename Scalar, size_t N>
constexpr Eigen::Matrix<Scalar, N, N> covariance_matrix(
    const std::array<Scalar, N>& std_devs) {
  Eigen::Matrix<Scalar, N, N> result;

  for (int row = 0; row < result.rows(); ++row) {
    for (int col = 0; col < result.cols(); ++col) {
      if (row == col) {
        result(row, col) = std_devs[row] * std_devs[row];
      } else {
        result(row, col) = Scalar(0);
      }
    }
  }

  return result;
}

/// Square-Root Kalman Filter.
///
/// [1] K. S. Tracy "A Square-Root Kalman Filter using only QR Decompositions",
///     URL: https://arxiv.org/pdf/2208.06452
///
/// @tparam Scalar Scalar type.
/// @tparam States Number of states.
/// @tparam Inputs Number of inputs.
/// @tparam Outputs Number of outputs.
template <typename Scalar, int States, int Inputs, int Outputs>
class SRKF {
 public:
  template <int Rows, int Cols>
  using DenseMatrix = Eigen::Matrix<Scalar, Rows, Cols>;

  template <int Rows>
  using DenseVector = Eigen::Vector<Scalar, Rows>;

  /// Constructs a Kalman filter with the given plant.
  ///
  /// The initial state estimate is zero.
  ///
  /// @param A Continuous system matrix.
  /// @param B Continuous input matrix.
  /// @param state_std_devs Continuous process noise standard deviations.
  /// @param measurement_std_devs Measurement standard deviations.
  /// @param initial_state_std_devs Standard deviations of the initial state
  ///     estimate. The initial error covariance is a diagonal matrix with these
  ///     values squared.
  SRKF(const DenseMatrix<States, States>& A,
       const DenseMatrix<States, Inputs>& B,
       const std::array<Scalar, States>& state_std_devs,
       const std::array<Scalar, Outputs>& measurement_std_devs,
       const std::array<Scalar, States>& initial_state_std_devs)
      : m_A{A},
        m_B{B},
        m_cont_Q{covariance_matrix(state_std_devs)},
        m_sr_R{covariance_matrix(measurement_std_devs).llt().matrixU()},
        m_x_hat{DenseVector<States>::Zero()},
        m_F{covariance_matrix(initial_state_std_devs).llt().matrixU()} {}

  /// Returns the state estimate x̂.
  const DenseVector<States>& x_hat() const { return m_x_hat; }

  /// Returns the error covariance P = FᵀF.
  DenseMatrix<States, States> P() const { return m_F.transpose() * m_F; }

  /// Project the model into the future with a new control input u.
  ///
  /// @param u New control input from controller.
  /// @param dt Timestep for prediction in seconds.
  void predict(const DenseVector<Inputs>& u, Scalar dt) {
    auto [A, B] = discretize_ab(m_A, m_B, dt);

    // √Q where Q = √Qᵀ√Q
    DenseMatrix<States, States> sr_Q =
        discretize_q(m_A, m_cont_Q, dt).llt().matrixU();

    // x̂ₖ₊₁|ₖ = Ax̂ₖ|ₖ + Buₖ
    m_x_hat = A * m_x_hat + B * u;

    // Fₖ₊₁|ₖ = qrᵣ([Fₖ|ₖAᵀ; √Q])
    m_F = qr_r<States, States, States>(m_F * A.transpose(), sr_Q);
  }

  /// Correct the state estimate x-hat using the measurements in y.
  ///
  /// @param u Same control input used in the predict step.
  /// @param y Measurement vector.
  /// @param C Output matrix.
  /// @param D Feedthrough matrix.
  void correct(const DenseVector<Inputs>& u, const DenseVector<Outputs>& y,
               const DenseMatrix<Outputs, States>& C,
               const DenseMatrix<Outputs, Inputs>& D) {
    // G = qrᵣ([Fₖ₊₁|ₖCᵀ; √R])
    DenseMatrix<Outputs, Outputs> G =
        qr_r<States, Outputs, Outputs>(m_F * C.transpose(), m_sr_R);

    // K = (G⁻¹(G⁻ᵀC)Fᵀₖ₊₁|ₖFₖ₊₁|ₖ)ᵀ
    // K = (G \ (Gᵀ \ C)Fᵀₖ₊₁|ₖFₖ₊₁|ₖ)ᵀ
    DenseMatrix<States, Outputs> K =
        (G.template triangularView<Eigen::Upper>().solve(
             G.template triangularView<Eigen::Upper>().transpose().solve(C)) *
         m_F.transpose() * m_F)
            .transpose();

    // x̂ₖ₊₁|ₖ₊₁ = x̂ₖ₊₁|ₖ + K(yₖ₊₁ − ŷₖ₊₁)
    m_x_hat += K * (y - (C * m_x_hat + D * u));

    // Fₖ₊₁|ₖ₊₁ = qrᵣ([Fₖ₊₁|ₖ(I − KC)ᵀ; √RKᵀ])
    m_F = qr_r<States, Outputs, States>(
        m_F * (DenseMatrix<States, States>::Identity() - K * C).transpose(),
        m_sr_R * K.transpose());
  }

 private:
  /// Continuous system matrix
  DenseMatrix<States, States> m_A;
  /// Continuous input matrix
  DenseMatrix<States, Inputs> m_B;
  /// Continuous process noise covariance matrix
  DenseMatrix<States, States> m_cont_Q;
  /// Square root of measurement noise covariance matrix
  DenseMatrix<Outputs, Outputs> m_sr_R;
  /// State estimate
  DenseVector<States> m_x_hat;
  /// Square root of error covariance stored in upper triangle
  DenseMatrix<States, States> m_F;

  /// Returns the upper triangular R from the QR decomposition of A stacked
  /// vertically on top of B.
  ///
  /// If M = [A; B], then RᵀR = MᵀM = AᵀA + BᵀB.
  ///
  /// @tparam Rows1 Rows of A.
  /// @tparam Rows2 Rows of B.
  /// @tparam Cols Columns of A and B.
  /// @param A The top matrix.
  /// @param B The bottom matrix.
  template <int Rows1, int Rows2, int Cols>
  static DenseMatrix<Cols, Cols> qr_r(const DenseMatrix<Rows1, Cols>& A,
                                      const DenseMatrix<Rows2, Cols>& B) {
    DenseMatrix<Rows1 + Rows2, Cols> M;
    M << A, B;
    return M.householderQr()
        .matrixQR()
        .template topRows<Cols>()
        .template triangularView<Eigen::Upper>();
  }
};

int main() {
  constexpr double kv = 0.2;
  constexpr double ka = 0.02;
  constexpr double dt = 0.05;

  constexpr Eigen::Matrix<double, 2, 2> A{{0.0, 1.0}, {0.0, -kv / ka}};
  constexpr Eigen::Matrix<double, 2, 1> B{{0.0}, {1.0 / ka}};
  constexpr Eigen::Matrix<double, 2, 2> C{{1.0, 0.0}, {0.0, 1.0}};
  constexpr Eigen::Matrix<double, 2, 1> D{{0.0}, {0.0}};

  SRKF<double, 2, 1, 2> kf{A, B, std::array{1.0, 2.0}, std::array{0.1, 0.01},
                           std::array{1.0, 2.0}};

  Eigen::Vector<double, 1> u{{12.0}};
  Eigen::Vector<double, 2> y{{1.0, 0.0}};

  kf.predict(u, dt);
  kf.correct(u, y, C, D);
}
