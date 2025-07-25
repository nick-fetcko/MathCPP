#pragma once

#include <algorithm>
#include <cstdint>

#include "Maths.hpp"

namespace MathsCPP {
template<typename T, std::size_t N, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class VectorBase {
protected:
	constexpr VectorBase() = default;
	template<typename ...Args>
	constexpr VectorBase(Args... args) : data{args...} {}
public:
	T data[N]{};
};

template<typename T>
class VectorBase<T, 1> {
protected:
	constexpr VectorBase() = default;
	constexpr VectorBase(T x) : x(x) {}
public:
	T x{};
};

template<typename T>
class VectorBase<T, 2> {
protected:
	constexpr VectorBase() = default;
	constexpr VectorBase(T x, T y) : x(x), y(y) {}
public:
	T x{}, y{};
};

template<typename T>
class VectorBase<T, 3> {
protected:
	constexpr VectorBase() = default;
	constexpr VectorBase(T x, T y, T z = 0) : x(x), y(y), z(z) {}
public:
	T x{}, y{}, z{};
};

template<typename T>
class VectorBase<T, 4> {
protected:
	constexpr VectorBase() = default;
	constexpr VectorBase(T x, T y, T z = 0, T w = 0) : x(x), y(y), z(z), w(w) {}
public:
	T x{}, y{}, z{}, w{};
};

/**
 * @brief Holds a N-tuple vector.
 * @tparam T The value type.
 * @tparam N Number of elements.
 */
template<typename T, std::size_t N>
class Vector : public VectorBase<T, N> {
public:
	constexpr Vector() = default;
	template<typename ...Args, typename = std::enable_if_t<sizeof...(Args) <= N && std::conjunction_v<std::is_arithmetic<Args>...>>>
	constexpr Vector(Args... args) : VectorBase<T, N>(static_cast<T>(args)...) {}
	
	template<typename T1, int ...S1>
	constexpr explicit Vector(T1 s, std::index_sequence<S1...>) : VectorBase<T, N>(s + (0 * S1)...) {}
	template<typename T1, typename = std::enable_if_t<std::is_arithmetic_v<T1>>>
	constexpr explicit Vector(T1 s) : Vector(s, std::make_index_sequence<N>()) {}

	template<typename T1, std::size_t N1, int ...S1, typename... Args>
	constexpr explicit Vector(const Vector<T1, N1> &v, std::index_sequence<S1...>, Args... args) : VectorBase<T, N>(v[S1]..., args...) {}
	template<typename T1, std::size_t N1, typename... Args, typename = std::enable_if_t<(N1 < N) && sizeof...(Args) == (N - N1)>>
	constexpr explicit Vector(const Vector<T1, N1> &v, Args... args) : Vector(v, std::make_index_sequence<N1>(), args...) {}

	template<typename T1, typename T2, std::size_t N1, std::size_t N2, int ...S1, int ...S2>
	constexpr Vector(const Vector<T1, N1> &v1, const Vector<T2, N2> &v2, std::index_sequence<S1...>, std::index_sequence<S2...>) : VectorBase<T, N>(v1[S1]..., v2[S2]...) {}
	template<typename T1, typename T2, std::size_t N1, std::size_t N2, typename = std::enable_if_t<N1 + N2 == N>>
	constexpr Vector(const Vector<T1, N1> &v1, const Vector<T2, N2> &v2) : Vector(v1, v2, std::make_index_sequence<N1>(), std::make_index_sequence<N2>()) {}

	template<typename T1, std::size_t N1, int ...S1, int ...S2>
	constexpr explicit Vector(const Vector<T1, N1> &v, std::index_sequence<S1...>, std::index_sequence<S2...>) : VectorBase<T, N>(v[S1]..., (0 * S2)...) {}
	template<typename T1, std::size_t N1, typename = std::enable_if_t<(N1 < N)>>
	constexpr explicit Vector(const Vector<T1, N1> &v) : Vector(v, std::make_index_sequence<N1>(), std::make_index_sequence<N - N1>()) {} // Vector(v, Vector<T1, N - N1>())

	//template<typename T1, std::size_t N1, int ...S1>
	//constexpr explicit Vector(const Vector<T1, N1> &v, std::index_sequence<S1...>) : VectorBase<T, N>(v[S1]...) {}
	//template<typename T1, std::size_t N1, typename = std::enable_if_t<N1 >= N>>
	//constexpr explicit Vector(const Vector<T1, N1> &v) : Vector(v, std::make_index_sequence<N>()) {}

	template<typename T1>
	constexpr Vector(const Vector<T1, N> &v) { copy_cast(v.begin(), v.end(), begin()); }

	template<typename T1>
	constexpr Vector &operator=(const Vector<T1, N> &v) {
		copy_cast(v.begin(), v.end(), begin());
		return *this;
	}

	constexpr const T &at(std::size_t i) const { return ((const T *)this)[i]; }
	constexpr T &at(std::size_t i) { return ((T *)this)[i]; }

	constexpr const T &operator[](std::size_t i) const { return at(i); }
	constexpr T &operator[](std::size_t i) { return at(i); }

	constexpr auto size() const { return N; }
	
	auto begin() { return &at(0); }
	auto begin() const { return &at(0); }

	auto end() { return &at(0) + N; }
	auto end() const { return &at(0) + N; }

	template<typename = std::enable_if_t<N >= 2>>
	constexpr const Vector<T, 2> &xy() const { return *reinterpret_cast<const Vector<T, 2> *>(this); }
	template<typename = std::enable_if_t<N >= 2>>
	constexpr Vector<T, 2> &xy() { return *reinterpret_cast<Vector<T, 2> *>(this); }
	
	template<typename = std::enable_if_t<N >= 3>>
	constexpr const Vector<T, 3> &xyz() const { return *reinterpret_cast<const Vector<T, 3> *>(this); }
	template<typename = std::enable_if_t<N >= 3>>
	constexpr Vector<T, 3> &xyz() { return *reinterpret_cast<Vector<T, 3> *>(this); }

	template<std::size_t ...I>
	constexpr auto Swizzle() const { return Vector<T, sizeof...(I)>(at(I)...); }

	/**
	 * Calculates the dot product of the this vector and another vector.
	 * @param other The other vector.
	 * @return The dot product.
	 */
	constexpr T Dot(const Vector &other) const {
		T result = 0;
		for (std::size_t i = 0; i < N; i++)
			result += at(i) * other[i];
		return result;
	}

	/**
	 * Gets the length squared of this vector.
	 * @return The length squared.
	 */
	constexpr T Length2() const {
		return Dot(*this);
	}
	
	/**
	 * Gets the length of this vector.
	 * @return The length.
	 */
	auto Length() const {
		return std::sqrt(Length2());
	}

	/**
	 * Gets the unit vector of this vector.
	 * @return The normalized vector.
	 */
	auto Normalize() const {
		return *this / Length();
	}

	/**
	 * Calculates the cross product of the this vector and another vector.
	 * @param other The other vector.
	 * @return The cross product.
	 */
	template<typename = std::enable_if_t<N == 2 || N == 3>>
	constexpr auto Cross(const Vector &other) const {
		if constexpr (N == 2) {
			return at(0) * other[1] - at(1) * other[0];
		} else if constexpr (N == 3) {
			return Vector(at(1) * other[2] - at(2) * other[1], at(2) * other[0] - at(0) * other[2], at(0) * other[1] - at(1) * other[0]);
		}
	}

	/**
	 * Gets the distance between this vector and another vector.
	 * @param other The other vector.
	 * @return The squared distance.
	 */
	constexpr T Distance2(const Vector &other) const {
		return (other - *this).Length2();
	}

	/**
	 * Gets the between this vector and another vector.
	 * @param other The other vector.
	 * @return The distance.
	 */
	auto Distance(const Vector &other) const {
		return (other - *this).Length();
	}

	/**
	 * Gets the vector distance between this vector and another vector.
	 * @param other The other vector.
	 * @return The vector distance.
	 */
	constexpr auto DistanceVector(const Vector &other) const {
		return (*this - other) * (*this - other);
	}

	/**
	 * Calculates the angle between this vector and another vector.
	 * @param other The other vector.
	 * @return The angle, in radians.
	 */
	T Uangle(const Vector &other) const {
		const T d = Dot(other);
		return d > 1 ? 0 : std::acos(d < -1 ? -1 : d);
	}
	
	/**
	 * Calculates the normalized angle between this vector and another vector.
	 * @param other The other vector.
	 * @return The angle, in radians.
	 */
	T Angle(const Vector &other) const {
		return Normalize().Uangle(other.Normalize());
	}

	template<typename T1>
	constexpr auto Lerp(const Vector &other, T1 c) const {
		return *this * (1 - c) + other * c;
	}
	
	template<typename T1>
	constexpr T Nlerp(const Vector &other, T1 t) const {
		return Lerp(other, t).Normalize();
	}
	
	template<typename T1, typename T2>
	T Slerp(const Vector<T1, N> &other, T2 t) const {
		T th = Uangle(other);
		return th == 0 ? *this : *this * (std::sin(th * (1 - t)) / std::sin(th)) + other * (std::sin(th * t) / std::sin(th));
	}

	/**
	 * Gets the absolute value of every component in this vector.
	 * @return The absolute value of this vector.
	 */
	Vector Abs() const {
		Vector result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = std::abs(at(i));
		return result;
	}
	
	/**
	 * Gets the minimal value in this vector.
	 * @return The minimal components.
	 */
	constexpr auto Min() const {
		return std::min(std::initializer_list(begin(), end()));
	}
	
	/**
	 * Gets the maximal value in this vector.
	 * @return The maximal components.
	 */
	constexpr auto Max() const {
		return std::max(std::initializer_list(begin(), end()));
	}
	
	/**
	 * Gets the minimal and maximal values in the vector.
	 * @return The minimal and maximal components.
	 */
	constexpr auto MinMax() const {
		return std::minmax(std::initializer_list(begin(), end()));
	}

	/**
	 * Gets the lowest vector size between this vector and other.
	 * @tparam T1 The others type.
	 * @param other The other vector to get values from.
	 * @return The lowest vector.
	 */
	template<typename T1>
	constexpr auto Min(const Vector<T1, N> &other) {
		using THighestP = decltype(at(0) + other[0]);
		Vector<THighestP, N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = std::min<THighestP>(at(i), other[1]);
		return result;
	}
	
	/**
	 * Gets the maximum vector size between this vector and other.
	 * @tparam T1 The others type.
	 * @param other The other vector to get values from.
	 * @return The maximum vector.
	 */
	template<typename T1>
	constexpr auto Max(const Vector<T1, N> &other) {
		using THighestP = decltype(at(0) + other[0]);
		Vector<THighestP, N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = std::max<THighestP>(at(i), other[1]);
		return result;
	}

	/**
	 * Rotates this vector by a angle around the origin.
	 * @tparam T1 The angle type.
	 * @param a The angle to rotate by, in radians.
	 * @return The rotated vector.
	 */
	template<typename T1, typename = std::enable_if_t<N == 2>>
	Vector Rotate(T1 a) const {
		const auto s = std::sin(a);
		const auto c = std::cos(a);
		return {at(0) * c - at(1) * s, at(0) * s + at(1) * c};
	}

	/**
	 * Gets if this vector is in a triangle.
	 * @param v1 The first triangle vertex.
	 * @param v2 The second triangle vertex.
	 * @param v3 The third triangle vertex.
	 * @return If this vector is in a triangle.
	 */
	template<typename = std::enable_if_t<N == 2>>
	constexpr bool InTriangle(const Vector &v1, const Vector &v2, const Vector &v3) const {
		auto b1 = ((at(0) - v2[0]) * (v1[1] - v2[1]) - (v1[0] - v2[1]) * (at(1) - v2[1])) < 0;
		auto b2 = ((at(0) - v3[0]) * (v2[1] - v3[1]) - (v2[0] - v3[1]) * (at(1) - v3[1])) < 0;
		auto b3 = ((at(0) - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[1]) * (at(1) - v1[1])) < 0;
		return ((b1 == b2) & (b2 == b3));
	}

	/**
	 * Converts from rectangular to spherical coordinates, this vector is in cartesian (x, y).
	 * @return The polar coordinates (radius, theta).
	 */
	template<typename = std::enable_if_t<N == 2 || N == 3>>
	auto CartesianToPolar() const {
		if constexpr (N == 2) {
			auto radius = std::sqrt(at(0) * at(0) + at(1) * at(1));
			auto theta = std::atan2(at(1), at(0));
			return Vector<decltype(radius), N>(radius, theta);
		} else if constexpr (N == 3) {
			auto radius = std::sqrt(at(0) * at(0) + at(1) * at(1) + at(2) * at(2));
			auto theta = std::atan2(at(1), at(0));
			auto phi = std::atan2(std::sqrt(at(0) * at(0) + at(1) * at(1)), at(2));
			return Vector<decltype(radius), N>(radius, theta, phi);
		}
	}
	
	/**
	 * Converts from spherical to rectangular coordinates, this vector is in polar (radius, theta).
	 * @return The cartesian coordinates (x, y).
	 */
	template<typename = std::enable_if_t<N == 2 || N == 3>>
	auto PolarToCartesian() const {
		if constexpr (N == 2) {
			auto x1 = at(0) * std::cos(at(1));
			auto y1 = at(0) * std::sin(at(0));
			return Vector<decltype(x1), N>(x1, y1);
		} else if constexpr (N == 3) {
			auto x1 = at(0) * std::sin(at(2)) * std::cos(at(1));
			auto y1 = at(0) * std::sin(at(2)) * std::sin(at(1));
			auto z1 = at(0) * std::cos(at(2));
			return Vector<decltype(x1), N>(x1, y1, z1);
		}
	}

	template<typename T1>
	constexpr friend auto operator==(const Vector &lhs, const Vector<T1, N> &rhs) {
		for (std::size_t i = 0; i < N; i++) {
			if (std::abs(lhs[i] - rhs[i]) > 0.0001f)
				return false;
		}
		return true;
	}

	template<typename T1>
	constexpr friend auto operator!=(const Vector &lhs, const Vector<T1, N> &rhs) {
		for (std::size_t i = 0; i < N; i++) {
			if (std::abs(lhs[i] - rhs[i]) > 0.0001f)
				return true;
		}
		return false;
	}

	constexpr friend auto operator+(const Vector &lhs) {
		Vector result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = +lhs[i];
		return result;
	}

	constexpr friend auto operator+(const Vector &lhs , const T &rhs) {
		Vector result = lhs;
		for (std::size_t i = 0; i < N; i++)
			result[i] += rhs;
		return result;
	}

	constexpr friend auto operator-(const Vector &lhs) {
		Vector result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = -lhs[i];
		return result;
	}

	constexpr friend auto operator-(const Vector &lhs, const T &rhs) {
		Vector result = lhs;
		for (std::size_t i = 0; i < N; i++)
			result[i] -= rhs;
		return result;
	}

	template<typename = std::enable_if_t<std::is_integral_v<T>>>
	constexpr friend auto operator~(const Vector &lhs) {
		Vector result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = ~lhs[i];
		return result;
	}

	template<typename = std::enable_if_t<std::is_integral_v<T>>>
	constexpr friend auto operator!(const Vector &lhs) {
		Vector result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = !lhs[i];
		return result;
	}

	template<typename T1>
	constexpr friend auto operator+(const Vector &lhs, const Vector<T1, N> &rhs) {
		Vector<decltype(lhs[0] + rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] + rhs[i];
		return result;
	}

	template<typename T1>
	constexpr friend auto operator-(const Vector &lhs, const Vector<T1, N> &rhs) {
		Vector<decltype(lhs[0] - rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] - rhs[i];
		return result;
	}

	constexpr friend auto operator*(const T &lhs, const Vector &rhs) {
		Vector result = rhs;
		for (std::size_t i = 0; i < N; i++)
			result[i] *= lhs;
		return result;
	}

	template<typename T1>
	constexpr friend auto operator*(const Vector &lhs, const Vector<T1, N> &rhs) {
		Vector<decltype(lhs[0] * rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] * rhs[i];
		return result;
	}

	template<typename T1>
	constexpr friend auto operator/(const Vector &lhs, const Vector<T1, N> &rhs) {
		Vector<decltype(lhs[0] / rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] / rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator%(const Vector &lhs, const Vector<T1, N> &rhs) {
		Vector<decltype(lhs[0] % rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] & rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator|(const Vector &lhs, const Vector<T1, N> &rhs) {
		Vector<decltype(lhs[0] | rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] | rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator^(const Vector &lhs, const Vector<T1, N> &rhs) {
		Vector<decltype(lhs[0] ^ rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] ^ rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator&(const Vector &lhs, const Vector<T1, N> &rhs) {
		Vector<decltype(lhs[0] & rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] & rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator<<(const Vector &lhs, const Vector<T1, N> &rhs) {
		Vector<decltype(lhs[0] << rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] << rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator>>(const Vector &lhs, const Vector<T1, N> &rhs) {
		Vector<decltype(lhs[0] >> rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] >> rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_arithmetic_v<T1>>>
	constexpr friend auto operator*(const Vector &lhs, T1 rhs) {
		Vector<decltype(lhs[0] * rhs), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] * rhs;
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_arithmetic_v<T1>>>
	constexpr friend auto operator/(const Vector &lhs, T1 rhs) {
		Vector<decltype(lhs[0] / rhs), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] / rhs;
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator%(const Vector &lhs, T1 rhs) {
		Vector<decltype(lhs[0] % rhs), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] & rhs;
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator|(const Vector &lhs, T1 rhs) {
		Vector<decltype(lhs[0] | rhs), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] | rhs;
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator^(const Vector &lhs, T1 rhs) {
		Vector<decltype(lhs[0] ^ rhs), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] ^ rhs;
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator&(const Vector &lhs, T1 rhs) {
		Vector<decltype(lhs[0] & rhs), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] & rhs;
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator<<(const Vector &lhs, T1 rhs) {
		Vector<decltype(lhs[0] << rhs), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] << rhs;
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator>>(const Vector &lhs, T1 rhs) {
		Vector<decltype(lhs[0] >> rhs), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs[i] >> rhs;
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_arithmetic_v<T1>>>
	constexpr friend auto operator*(T1 lhs, const Vector &rhs) {
		Vector<decltype(lhs *rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs * rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_arithmetic_v<T1>>>
	constexpr friend auto operator/(T1 lhs, const Vector &rhs) {
		Vector<decltype(lhs / rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs / rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator%(T1 lhs, const Vector &rhs) {
		Vector<decltype(lhs %rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs & rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator|(T1 lhs, const Vector &rhs) {
		Vector<decltype(lhs | rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs | rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator^(T1 lhs, const Vector &rhs) {
		Vector<decltype(lhs ^rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs ^ rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator&(T1 lhs, const Vector &rhs) {
		Vector<decltype(lhs &rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs & rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator<<(T1 lhs, const Vector &rhs) {
		Vector<decltype(lhs << rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs << rhs[i];
		return result;
	}

	template<typename T1, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<T1>>>
	constexpr friend auto operator>>(T1 lhs, const Vector &rhs) {
		Vector<decltype(lhs >> rhs[0]), N> result;
		for (std::size_t i = 0; i < N; i++)
			result[i] = lhs >> rhs[i];
		return result;
	}

	template<typename T1>
	constexpr friend auto operator+=(Vector &lhs, const T1 &rhs) {
		return lhs = lhs + rhs;
	}

	template<typename T1>
	constexpr friend auto operator-=(Vector &lhs, const T1 &rhs) {
		return lhs = lhs - rhs;
	}

	template<typename T1>
	constexpr friend auto operator*=(Vector &lhs, const T1 &rhs) {
		return lhs = lhs * rhs;
	}

	template<typename T1>
	constexpr friend auto operator/=(Vector &lhs, const T1 &rhs) {
		return lhs = lhs / rhs;
	}

	template<typename T1>
	constexpr friend auto operator%=(Vector &lhs, const T1 &rhs) {
		return lhs = lhs % rhs;
	}

	template<typename T1>
	constexpr friend auto operator|=(Vector &lhs, const T1 &rhs) {
		return lhs = lhs | rhs;
	}

	template<typename T1>
	constexpr friend auto operator^=(Vector &lhs, const T1 &rhs) {
		return lhs = lhs ^ rhs;
	}

	template<typename T1>
	constexpr friend auto operator&=(Vector &lhs, const T1 &rhs) {
		return lhs = lhs & rhs;
	}

	template<typename T1>
	constexpr friend auto operator<<=(Vector &lhs, const T1 &rhs) {
		return lhs = lhs << rhs;
	}

	template<typename T1>
	constexpr friend auto operator>>=(Vector &lhs, const T1 &rhs) {
		return lhs = lhs >> rhs;
	}

	friend std::ostream &operator<<(std::ostream &stream, const Vector &vector) {
		for (std::size_t i = 0; i < N; i++)
			stream << vector[i] << (i != N - 1 ? ", " : "");
		return stream;
	}
	
	static const Vector Zero;
	static const Vector One;
	static const Vector Infinity;
	static const Vector Right;
	static const Vector Left;
	static const Vector Up;
	static const Vector Down;
	static const Vector Front;
	static const Vector Back;
};

template<typename T, std::size_t N>
const Vector<T, N> Vector<T, N>::Zero = Vector<T, N>(0);
template<typename T, std::size_t N>
const Vector<T, N> Vector<T, N>::One = Vector<T, N>(1);
template<typename T, std::size_t N>
const Vector<T, N> Vector<T, N>::Infinity = Vector<T, N>(std::numeric_limits<T>::infinity());
template<typename T, std::size_t N> 
const Vector<T, N> Vector<T, N>::Right = Vector<T, N>(1, 0);
template<typename T, std::size_t N>
const Vector<T, N> Vector<T, N>::Left = Vector<T, N>(-1, 0);
template<typename T, std::size_t N>
const Vector<T, N> Vector<T, N>::Up = Vector<T, N>(0, 1);
template<typename T, std::size_t N>
const Vector<T, N> Vector<T, N>::Down = Vector<T, N>(0, -1);
template<typename T, std::size_t N>
const Vector<T, N> Vector<T, N>::Front = Vector<T, N>(0, 0, 1);
template<typename T, std::size_t N>
const Vector<T, N> Vector<T, N>::Back = Vector<T, N>(0, 0, -1);

using Vector1f = Vector<float, 1>;
using Vector1d = Vector<double, 1>;
using Vector1i = Vector<int32_t, 1>;
using Vector1ui = Vector<uint32_t, 1>;

using Vector2f = Vector<float, 2>;
using Vector2d = Vector<double, 2>;
using Vector2i = Vector<int32_t, 2>;
using Vector2ui = Vector<uint32_t, 2>;

using Vector3f = Vector<float, 3>;
using Vector3d = Vector<double, 3>;
using Vector3i = Vector<int32_t, 3>;
using Vector3ui = Vector<uint32_t, 3>;

using Vector4f = Vector<float, 4>;
using Vector4d = Vector<double, 4>;
using Vector4i = Vector<int32_t, 4>;
using Vector4ui = Vector<uint32_t, 4>;
}

namespace std {
template<typename T, size_t N>
struct hash<MathsCPP::Vector<T, N>> {
	size_t operator()(const MathsCPP::Vector<T, N> &vector) const noexcept {
		size_t seed = 0;
		for (size_t i = 0; i < N; i++)
			MathsCPP::Maths::HashCombine(seed, vector[i]);
		return seed;
	}
};
}
