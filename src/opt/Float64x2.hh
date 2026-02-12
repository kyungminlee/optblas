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

struct Float64x2 {
  double hi;
  double lo;

  // Constructors
  constexpr Float64x2() : hi(0.0), lo(0.0) {}
  constexpr Float64x2(double h, double l) : hi(h), lo(l) {}

  // Implicit conversion from double
  constexpr Float64x2(double d) : hi(d), lo(0.0) {}

  // Unary Negation
  constexpr Float64x2 operator-() const { return Float64x2(-hi, -lo); }

  // Renormalization helper (Hida/Bailey style)
  // Ensures |lo| <= 0.5 * eps(hi)
  static Float64x2 normalize(double h, double l) {
    auto res = quick_two_sum(h, l);
    return Float64x2(res.first, res.second);
  }

  // --- Arithmetic Operators ---

  // Addition: (a + b)
  // Algorithm: Hida, Li, Bailey (2000)
  friend Float64x2 operator+(const Float64x2 &a, const Float64x2 &b) {
    auto s = two_sum(a.hi, b.hi);
    auto t = two_sum(a.lo, b.lo);
    s.second += s.first; // This line seems wrong in standard alg, let's use the
                         // explicit robust steps below:

    // Correct robust Double-Double Addition (slow-two-sum style for max
    // precision)
    // 1. Add highs
    auto s1 = two_sum(a.hi, b.hi);
    // 2. Add lows
    auto s2 = two_sum(a.lo, b.lo);

    // 3. Mix terms
    s1.second += s2.first;
    auto s3 = quick_two_sum(s1.first, s1.second);
    s3.second += s2.second;

    return normalize(s3.first, s3.second);
  }

  Float64x2 & operator+=(Float64x2 const & rhs) {
    (*this) = (*this) + rhs;
    return *this;
  }

  Float64x2 & operator*=(Float64x2 const & rhs) {
    (*this) = (*this) * rhs;
    return *this;
  }

  // Subtraction: (a - b)
  friend Float64x2 operator-(const Float64x2 &a, const Float64x2 &b) {
    return a + (-b);
  }

  // Multiplication: (a * b)
  // Algorithm: Hida, Li, Bailey (2000)
  // We ignore the O(eps^2) term (a.lo * b.lo) as it falls below the precision
  // of Float64x2.
  friend Float64x2 operator*(const Float64x2 &a, const Float64x2 &b) {
    auto p = two_prod(a.hi, b.hi);

    // Cross terms
    double t = a.hi * b.lo + a.lo * b.hi;

    // Inner add
    p.second += t;

    return normalize(p.first, p.second);
  }

  // Mixed Multiplication: (DoubleDouble * double)
  friend Float64x2 operator*(const Float64x2 &a, double b) {
    auto p = two_prod(a.hi, b);
    double t = a.lo * b;
    p.second += t;
    return normalize(p.first, p.second);
  }

  // Mixed Multiplication: (double * DoubleDouble)
  friend Float64x2 operator*(double a, const Float64x2 &b) { return b * a; }

  // Mixed Addition: (DoubleDouble + double)
  friend Float64x2 operator+(const Float64x2 &a, double b) {
    auto s = two_sum(a.hi, b);
    s.second += a.lo;
    return normalize(s.first, s.second);
  }

  // Mixed Addition: (double + DoubleDouble)
  friend Float64x2 operator+(double a, const Float64x2 &b) { return b + a; }

  // Mixed Subtraction: (DoubleDouble - double)
  friend Float64x2 operator-(const Float64x2 &a, double b) { return a + (-b); }

  // Mixed Subtraction: (double - DoubleDouble)
  friend Float64x2 operator-(double a, const Float64x2 &b) {
    // a - (hi + lo) = (a - hi) - lo
    auto s = two_sum(a, -b.hi);
    s.second -= b.lo;
    return normalize(s.first, s.second);
  }

  // --- Comparison Operators ---

  // Equality
  // MultiFloats.jl generally keeps numbers normalized, so strict component
  // equality works.
  friend bool operator==(const Float64x2 &a, const Float64x2 &b) {
    return a.hi == b.hi && a.lo == b.lo;
  }

  friend bool operator!=(const Float64x2 &a, const Float64x2 &b) {
    return !(a == b);
  }

  // Less Than (Lexicographical for normalized numbers)
  friend bool operator<(const Float64x2 &a, const Float64x2 &b) {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
  }

  friend bool operator>(const Float64x2 &a, const Float64x2 &b) {
    return b < a;
  }

  friend bool operator<=(const Float64x2 &a, const Float64x2 &b) {
    return !(b < a);
  }

  friend bool operator>=(const Float64x2 &a, const Float64x2 &b) {
    return !(a < b);
  }

  // Output stream helper
  friend std::ostream &operator<<(std::ostream &os, const Float64x2 &val) {
    os << "(" << val.hi << ", " << val.lo << ")";
    return os;
  }
};

