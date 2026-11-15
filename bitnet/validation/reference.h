#pragma once

#include "contract.h"
#include <array>
#include <cstdint>
#include <vector>

struct Input {
    std::vector<uint8_t> weights;
    std::array<float, TQ2_COLUMNS> activation{};
};

Input known_answer_input();
Input dense_input();
std::array<double, TQ2_ROWS> reference_matvec(const Input &input);

// Independent logical-column decoder for GGML dequantization cross-check only.
double decoded_weight(const Input &input, size_t row, size_t column);
