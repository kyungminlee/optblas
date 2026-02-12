#include <cmath>
#include <iostream>
#include <limits>
#include <utility>

// Error-Free Transformation: TwoSum
// Computes s = a + b and error e such that s + e = a + b exactly.
// This algorithm (Knuth) works even if |a| < |b|.
inline std::pair<double, double> two_sum(double a, double b) {
  double s = a + b;
  double v = s - a;
  double e = (a - (s - v)) + (b - v);
  return {s, e};
}

// Error-Free Transformation: QuickTwoSum
// Computes s = a + b and error e, assuming |a| >= |b|.
inline std::pair<double, double> quick_two_sum(double a, double b) {
  double s = a + b;
  double e = b - (s - a);
  return {s, e};
}

// Error-Free Transformation: TwoProd
// Computes p = a * b and error e such that p + e = a * b exactly.
// Uses Fused Multiply-Add (FMA) for performance and accuracy.
inline std::pair<double, double> two_prod(double a, double b) {
  double p = a * b;
  double e = std::fma(a, b, -p);
  return {p, e};
}

struct float64x2 {
  double limbs[2];

  // Constructors
  constexpr float64x2() : limbs{0.0, 0.0} {}
  constexpr float64x2(double h, double l) : limbs{h, l} {}

  // Implicit conversion from double
  constexpr float64x2(double d) : limbs{d, 0.0} {}

  // Unary Negation
  constexpr float64x2 operator-() const { return float64x2(-limbs[0], -limbs[1]); }

  // Renormalization helper (Hida/Bailey style)
  // Ensures |limbs[1]| <= 0.5 * eps(limbs[0])
  static float64x2 normalize(double h, double l) {
    auto res = quick_two_sum(h, l);
    return float64x2(res.first, res.second);
  }

  // --- Arithmetic Operators ---

  // Addition: (a + b)
  // Algorithm: Hida, Li, Bailey (2000)
  friend float64x2 operator+(const float64x2 &a, const float64x2 &b) {
    auto s = two_sum(a.limbs[0], b.limbs[0]);
    auto t = two_sum(a.limbs[1], b.limbs[1]);
    s.second += s.first; // This line seems wrong in standard alg, let's use the
                         // explicit robust steps below:

    // Correct robust Double-Double Addition (slow-two-sum style for max
    // precision)
    // 1. Add highs
    auto s1 = two_sum(a.limbs[0], b.limbs[0]);
    // 2. Add lows
    auto s2 = two_sum(a.limbs[1], b.limbs[1]);

    // 3. Mix terms
    s1.second += s2.first;
    auto s3 = quick_two_sum(s1.first, s1.second);
    s3.second += s2.second;

    return normalize(s3.first, s3.second);
  }

  float64x2 & operator+=(float64x2 const & rhs) {
    (*this) = (*this) + rhs;
    return *this;
  }

  float64x2 & operator*=(float64x2 const & rhs) {
    (*this) = (*this) * rhs;
    return *this;
  }

  // Subtraction: (a - b)
  friend float64x2 operator-(const float64x2 &a, const float64x2 &b) {
    return a + (-b);
  }

  // Multiplication: (a * b)
  // Algorithm: Hida, Li, Bailey (2000)
  // We ignore the O(eps^2) term (a.limbs[1] * b.limbs[1]) as it falls below the precision
  // of Float64x2.
  friend float64x2 operator*(const float64x2 &a, const float64x2 &b) {
    auto p = two_prod(a.limbs[0], b.limbs[0]);

    // Cross terms
    double t = a.limbs[0] * b.limbs[1] + a.limbs[1] * b.limbs[0];

    // Inner add
    p.second += t;

    return normalize(p.first, p.second);
  }

  // Mixed Multiplication: (DoubleDouble * double)
  friend float64x2 operator*(const float64x2 &a, double b) {
    auto p = two_prod(a.limbs[0], b);
    double t = a.limbs[1] * b;
    p.second += t;
    return normalize(p.first, p.second);
  }

  // Mixed Multiplication: (double * DoubleDouble)
  friend float64x2 operator*(double a, const float64x2 &b) { return b * a; }

  // Mixed Addition: (DoubleDouble + double)
  friend float64x2 operator+(const float64x2 &a, double b) {
    auto s = two_sum(a.limbs[0], b);
    s.second += a.limbs[1];
    return normalize(s.first, s.second);
  }

  // Mixed Addition: (double + DoubleDouble)
  friend float64x2 operator+(double a, const float64x2 &b) { return b + a; }

  // Mixed Subtraction: (DoubleDouble - double)
  friend float64x2 operator-(const float64x2 &a, double b) { return a + (-b); }

  // Mixed Subtraction: (double - DoubleDouble)
  friend float64x2 operator-(double a, const float64x2 &b) {
    // a - (limbs[0] + limbs[1]) = (a - limbs[0]) - limbs[1]
    auto s = two_sum(a, -b.limbs[0]);
    s.second -= b.limbs[1];
    return normalize(s.first, s.second);
  }

  // --- Comparison Operators ---

  // Equality
  // MultiFloats.jl generally keeps numbers normalized, so strict component
  // equality works.
  friend bool operator==(const float64x2 &a, const float64x2 &b) {
    return a.limbs[0] == b.limbs[0] && a.limbs[1] == b.limbs[1];
  }

  friend bool operator!=(const float64x2 &a, const float64x2 &b) {
    return !(a == b);
  }

  // Less Than (Lexicographical for normalized numbers)
  friend bool operator<(const float64x2 &a, const float64x2 &b) {
    return (a.limbs[0] < b.limbs[0]) || (a.limbs[0] == b.limbs[0] && a.limbs[1] < b.limbs[1]);
  }

  friend bool operator>(const float64x2 &a, const float64x2 &b) {
    return b < a;
  }

  friend bool operator<=(const float64x2 &a, const float64x2 &b) {
    return !(b < a);
  }

  friend bool operator>=(const float64x2 &a, const float64x2 &b) {
    return !(a < b);
  }

  // Output stream helper
  friend std::ostream &operator<<(std::ostream &os, const float64x2 &val) {
    os << "(" << val.limbs[0] << ", " << val.limbs[1] << ")";
    return os;
  }
};

#ifdef __cplusplus
#pragma omp declare reduction(+ : float64x2 : omp_out += omp_in) initializer(omp_priv=float64x2(0.0))
#endif
