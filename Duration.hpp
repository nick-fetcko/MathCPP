﻿#pragma once

#include <chrono>
#include <algorithm>

#include "Maths.hpp"

namespace MathsCPP {
using namespace std::chrono_literals;

template<class T>
struct is_duration : std::false_type {};

template<class Rep, class Period>
struct is_duration<std::chrono::duration<Rep, Period>> : std::true_type {};

template<class T>
inline constexpr bool is_duration_v = is_duration<T>::value;

template<typename T, typename = std::enable_if_t<is_duration_v<T>>>
class Duration {
public:
	constexpr Duration() = default;
	template<typename Rep, typename Period>
	constexpr Duration(const std::chrono::duration<Rep, Period> &d) : value(std::chrono::duration_cast<T>(d).count()) {}
	template<typename T1>
	constexpr Duration(const Duration<T1> &d) : value(std::chrono::duration_cast<T>(d.value).count()) {}

	template<typename Rep, typename Period>
	constexpr Duration &operator=(const std::chrono::duration<Rep, Period> &d) {
		value = {std::chrono::duration_cast<T>(d).count()};
		return *this;
	}

	template<typename T1>
	constexpr Duration &operator=(const Duration<T1> &d) {
		value = {std::chrono::duration_cast<T>(d.value).count()};
		return *this;
	}

	template<typename T1, typename T2 = typename T1::rep, typename = std::enable_if_t<is_duration_v<T1>>>
	constexpr auto Cast() const {
		return static_cast<T2>(value.count()) / static_cast<T2>(typename std::ratio_divide<typename T::period, typename T1::period>::den);
	}

	constexpr auto AsSeconds() const {
		return value / 1.0s;
	}

	static Duration Now() {
		static const auto LocalEpoch = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<T>(std::chrono::high_resolution_clock::now() - LocalEpoch);

		//auto now = std::chrono::system_clock::now();
		//return std::chrono::duration_cast<T>(now.time_since_epoch());
	}

	static std::string GetDateTime(const std::string &format = "%Y-%m-%d %H:%M:%S") {
		auto now = std::chrono::system_clock::now();
		auto timeT = std::chrono::system_clock::to_time_t(now);

		std::stringstream ss;
		ss << std::put_time(std::localtime(&timeT), format.c_str());
		return ss.str();
	}

	template<typename Rep, typename Period>
	constexpr operator std::chrono::duration<Rep, Period>() const {
		return std::chrono::duration_cast<std::chrono::duration<Rep, Period>>(value);
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_convertible_v<T1, T>>>
	constexpr friend auto operator==(const Duration &lhs, const T1 &rhs) {
		return lhs.value == Duration<T>(rhs).value;
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_convertible_v<T1, T>>>
	constexpr friend auto operator!=(const Duration &lhs, const T1 &rhs) {
		return lhs.value != Duration<T>(rhs).value;
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_convertible_v<T1, T>>>
	constexpr friend auto operator<(const Duration &lhs, const T1 &rhs) {
		return lhs.value < Duration<T>(rhs).value;
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_convertible_v<T1, T>>>
	constexpr friend auto operator<=(const Duration &lhs, const T1 &rhs) {
		return lhs.value <= Duration<T>(rhs).value;
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_convertible_v<T1, T>>>
	constexpr friend auto operator>(const Duration &lhs, const T1 &rhs) {
		return lhs.value > Duration<T>(rhs).value;
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_convertible_v<T1, T>>>
	constexpr friend auto operator>=(const Duration &lhs, const T1 &rhs) {
		return lhs.value >= Duration<T>(rhs).value;
	}

	template<typename T1>
	constexpr friend auto operator+(const Duration &lhs, const Duration<T1> &rhs) {
		return Duration(lhs.value + rhs.value);
	}

	template<typename T1>
	constexpr friend auto operator-(const Duration &lhs, const Duration<T1> &rhs) {
		return Duration(lhs.value - rhs.value);
	}

	template<typename T1>
	constexpr friend auto operator*(const Duration &lhs, const Duration<T1> &rhs) {
		return Duration(lhs.value * rhs.value);
	}

	template<typename T1>
	constexpr friend auto operator/(const Duration &lhs, const Duration<T1> &rhs) {
		return Duration(lhs.value / rhs.value);
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1>>>
	constexpr friend auto operator+(const Duration &lhs, T1 rhs) {
		return Duration(lhs.value + rhs);
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1>>>
	constexpr friend auto operator-(const Duration &lhs, T1 rhs) {
		return Duration(lhs.value - rhs);
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_arithmetic_v<T1>>>
	constexpr friend auto operator*(const Duration &lhs, T1 rhs) {
		return Duration(lhs.value * rhs);
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_arithmetic_v<T1>>>
	constexpr friend auto operator/(const Duration &lhs, T1 rhs) {
		return Duration(lhs.value / rhs);
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_integral_v<T1>>>
	constexpr friend auto operator%(const Duration &lhs, T1 rhs) {
		return lhs.value % rhs;
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1>>>
	constexpr friend auto operator+(T1 lhs, const Duration &rhs) {
		return Duration(lhs + rhs.value);
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1>>>
	constexpr friend auto operator-(T1 lhs, const Duration &rhs) {
		return Duration(lhs - rhs.value);
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_arithmetic_v<T1>>>
	constexpr friend auto operator*(T1 lhs, const Duration &rhs) {
		return Duration(lhs * rhs.value);
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_arithmetic_v<T1>>>
	constexpr friend auto operator/(T1 lhs, const Duration &rhs) {
		return Duration(lhs / rhs.value);
	}

	template<typename T1, typename = std::enable_if_t<is_duration_v<T1> || std::is_integral_v<T1>>>
	constexpr friend auto operator%(T1 lhs, const Duration &rhs) {
		return lhs % rhs.value;
	}

	template<typename T1>
	constexpr friend auto operator+=(Duration &lhs, const T1 &rhs) {
		return lhs = lhs + rhs;
	}

	template<typename T1>
	constexpr friend auto operator-=(Duration &lhs, const T1 &rhs) {
		return lhs = lhs - rhs;
	}

	template<typename T1>
	constexpr friend auto operator*=(Duration &lhs, const T1 &rhs) {
		return lhs = lhs * rhs;
	}

	template<typename T1>
	constexpr friend auto operator/=(Duration &lhs, const T1 &rhs) {
		return lhs = lhs / rhs;
	}

	T value{};
};

using Nanoseconds = std::chrono::nanoseconds;
using Microseconds = std::chrono::microseconds;
using Milliseconds = std::chrono::milliseconds;
using Seconds = std::chrono::seconds;
using Minutes = std::chrono::minutes;
using Hours = std::chrono::hours;

//using Durationn = Duration<Nanoseconds>;
//using Durationu = Duration<Microseconds>;
//using Durationm = Duration<Milliseconds>;
//using Durations = Duration<Seconds>;

class Delta {
public:
	const Delta &Update() {
		current = Duration<Microseconds>::Now();
		change = current - last;
		last = current;

		return *this;
	}

	Duration<Microseconds> change;

private:
	Duration<Microseconds> current;
	Duration<Microseconds> last;
};

class Decay {
public:
	void SetTime(Duration<Microseconds> time) { this->time = time; decay = time; }
	const float &Get() const { return value; }

	void Update(const Delta &delta) {
		decay -= delta.change;
		sinceLastReset += delta.change;
		value = std::max(
			0.0f,
			static_cast<float>(decay.value.count()) / time.value.count() * maxValue
		);
	}

	void Reset(float value) { 
		this->value = value; 
		maxValue = value;
		decay = time; 
		wasReset = true;
		sinceLastReset.value = 0us;
	}

	bool WasReset() const { return wasReset; }
	void HasBeenReset() { this->wasReset = false; }

	const Duration<Microseconds> &SinceLastReset() const { return sinceLastReset; }

private:
	Duration<Microseconds> decay;
	Duration<Microseconds> time;
	Duration<Microseconds> sinceLastReset = 0us;

	float value = 0.0f;
	float maxValue = 0.0f;
	bool wasReset = false;
};
}
