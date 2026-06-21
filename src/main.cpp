#include <array>

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
/// @param disc_A Storage for discrete system matrix.
/// @param disc_B Storage for discrete input matrix.
template <typename Scalar, int States, int Inputs>
void discretize_ab(const Eigen::Matrix<Scalar, States, States>& cont_A,
                   const Eigen::Matrix<Scalar, States, Inputs>& cont_B,
                   double dt, Eigen::Matrix<Scalar, States, States>* disc_A,
                   Eigen::Matrix<Scalar, States, Inputs>* disc_B) {
  // M = [A  B]
  //     [0  0]
  Eigen::Matrix<Scalar, States + Inputs, States + Inputs> M;
  M.template block<States, States>(0, 0) = cont_A;
  M.template block<States, Inputs>(0, States) = cont_B;
  M.template block<Inputs, States + Inputs>(States, 0).setZero();

  // ϕ = eᴹᵀ = [A_d  B_d]
  //           [ 0    I ]
  Eigen::Matrix<Scalar, States + Inputs, States + Inputs> phi = (M * dt).exp();

  *disc_A = phi.template block<States, States>(0, 0);
  *disc_B = phi.template block<States, Inputs>(0, States);
}

/// Creates a covariance matrix from the given vector for use with Kalman
/// filters.
///
/// Each element is squared and placed on the covariance matrix diagonal.
///
/// @param std_devs An array. For a Q matrix, its elements are the standard
///     deviations of each state from how the model behaves. For an R matrix,
///     its elements are the standard deviations for each output measurement.
/// @return Process noise or measurement noise covariance matrix.
template <size_t N>
constexpr Eigen::Matrix<double, N, N> covariance_matrix(
    const std::array<double, N>& std_devs) {
  Eigen::Matrix<double, N, N> result;

  for (int row = 0; row < result.rows(); ++row) {
    for (int col = 0; col < result.cols(); ++col) {
      if (row == col) {
        result(row, col) = std_devs[row] * std_devs[row];
      } else {
        result(row, col) = 0.0;
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
  /// @param A System matrix.
  /// @param B Input matrix.
  /// @param state_std_devs State standard deviations.
  /// @param measurement_std_devs Measurement standard deviations.
  SRKF(const DenseMatrix<States, States>& A,
       const DenseMatrix<States, Inputs>& B,
       const std::array<Scalar, States>& state_std_devs,
       const std::array<Scalar, Outputs>& measurement_std_devs)
      : m_A{A},
        m_B{B},
        m_sr_Q{covariance_matrix(state_std_devs).llt().matrixU()},
        m_sr_R{covariance_matrix(measurement_std_devs).llt().matrixU()},
        m_x_hat{DenseVector<States>::Zero()},
        m_F{m_sr_Q} {}

  /// Project the model into the future with a new control input u.
  ///
  /// @param u New control input from controller.
  /// @param dt Timestep for prediction in seconds.
  void predict(const DenseVector<Inputs>& u, double dt) {
    DenseMatrix<States, States> A;
    DenseMatrix<States, Inputs> B;
    discretize_ab(m_A, m_B, dt, &A, &B);

    // x̂ₖ₊₁|ₖ = Ax̂ₖ|ₖ + Buₖ
    m_x_hat = A * m_x_hat + B * u;

    // Fₖ₊₁|ₖ = qrᵣ(Fₖ|ₖAᵀ, √Q)
    m_F = qr_r<States>(m_F * A.transpose(), m_sr_Q);
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
    // G = qrᵣ(Fₖ₊₁|ₖCᵀ, √R)
    DenseMatrix<Outputs, Outputs> G =
        qr_r<Outputs>(m_F * C.transpose(), m_sr_R);

    // K = (G⁻¹(G⁻ᵀC)Fᵀₖ₊₁|ₖFₖ₊₁|ₖ))ᵀ
    // K = (G \ (Gᵀ \ C) Fᵀₖ₊₁|ₖFₖ₊₁|ₖ)ᵀ
    DenseMatrix<States, Outputs> K =
        (G.template triangularView<Eigen::Upper>().solve(
             G.template triangularView<Eigen::Upper>().transpose().solve(C)) *
         m_F.transpose() * m_F)
            .transpose();

    // x̂ₖ₊₁|ₖ₊₁ = x̂ₖ₊₁|ₖ + K(yₖ₊₁ − ŷₖ₊₁)
    m_x_hat += K * (y - (C * m_x_hat + D * u));

    // Fₖ₊₁|ₖ₊₁ = qrᵣ(Fₖ₊₁|ₖ(I − KC)ᵀ, √RKᵀ)
    m_F = qr_r<Outputs>(
        m_F * (DenseMatrix<States, States>::Identity() - K * C).transpose(),
        m_sr_R * K.transpose());
  }

 private:
  /// Continuous system matrix
  DenseMatrix<States, States> m_A;
  /// Continuous input matrix
  DenseMatrix<States, Inputs> m_B;
  /// Square root of discrete process noise covariance matrix
  DenseMatrix<States, States> m_sr_Q;
  /// Square root of discrete measurement noise covariance matrix
  DenseMatrix<Outputs, Outputs> m_sr_R;
  /// State estimate
  DenseVector<States> m_x_hat;
  /// Square root of error covariance
  DenseMatrix<States, States> m_F;

  /// Returns R from the QR decomposition of [A, B].
  ///
  /// @tparam Cols Columns of B
  /// @param A The left matrix.
  /// @param B The right matrix.
  template <int Cols>
  DenseMatrix<States, States> qr_r(const DenseMatrix<States, Cols>& A,
                                   const DenseMatrix<Cols, Cols>& B) {
    return (DenseMatrix<States, States + Cols>{} << A, B)
        .finished()
        .householderQr()
        .matrixQR()
        .template block<Cols, Cols>(0, 0)
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

  SRKF<double, 2, 1, 2> kf{A, B, std::array{1.0, 2.0}, std::array{0.1, 0.01}};

  Eigen::Vector<double, 1> u{{12.0}};
  Eigen::Vector<double, 2> y{{1.0, 0.0}};

  kf.predict(u, dt);
  kf.correct(u, y, C, D);
}
