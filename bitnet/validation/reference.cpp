#include "reference.h"

#include <cstring>
#include <random>

namespace {
constexpr size_t weight_bytes = TQ2_ROWS * TQ2_BLOCKS_PER_ROW * TQ2_BLOCK_BYTES;
static_assert(sizeof(_Float16) == 2);

void set_scale(uint8_t *block, float value) {
    const _Float16 half_scale = static_cast<_Float16>(value);
    uint16_t bits;
    std::memcpy(&bits, &half_scale, sizeof(bits));
    block[TQ2_CODES_BYTES] = bits & 0xff;
    block[TQ2_CODES_BYTES + 1] = bits >> 8;
}

double read_scale(const uint8_t *block) {
    const uint16_t bits = block[TQ2_CODES_BYTES] | (uint16_t(block[TQ2_CODES_BYTES + 1]) << 8);
    _Float16 half_scale;
    std::memcpy(&half_scale, &bits, sizeof(bits));
    return static_cast<double>(half_scale);
}
}

Input known_answer_input() {
    Input input;
    input.weights.resize(weight_bytes);
    for (size_t offset = 0; offset < weight_bytes; offset += TQ2_BLOCK_BYTES) {
        std::memset(input.weights.data() + offset, 0x55, TQ2_CODES_BYTES);
        set_scale(input.weights.data() + offset, 1.0f);
    }
    // Row 0, block 0: codes [0,1,2,0] at logical columns [3,35,67,99].
    input.weights[3] = 0x24;
    set_scale(input.weights.data(), 0.5f);
    input.activation[3] = 1.25f;
    input.activation[35] = 4.0f;
    input.activation[67] = 2.0f;
    input.activation[99] = -0.5f;

    // Row 0, block 1: column 385 is the first plane of the second half.
    input.weights[TQ2_BLOCK_BYTES + 33] = 0x56;
    set_scale(input.weights.data() + TQ2_BLOCK_BYTES, 0.25f);
    input.activation[385] = 3.0f;

    // A nonzero code with zero scale and a nonzero final row.
    uint8_t *second_row = input.weights.data() + TQ2_BLOCKS_PER_ROW * TQ2_BLOCK_BYTES;
    second_row[3] = 0x56;
    set_scale(second_row, 0.0f);
    uint8_t *last_row = input.weights.data() + (TQ2_ROWS - 1) * TQ2_BLOCKS_PER_ROW * TQ2_BLOCK_BYTES;
    last_row[3] = 0x54;
    return input;
}

Input dense_input() {
    Input input;
    input.weights.resize(weight_bytes);
    std::mt19937 random(42);
    for (float &value : input.activation) {
        value = static_cast<float>(random() >> 8) / 8388608.0f - 1.0f;
    }
    for (size_t offset = 0; offset < weight_bytes; offset += TQ2_BLOCK_BYTES) {
        uint8_t *block = input.weights.data() + offset;
        for (int j = 0; j < TQ2_CODES_BYTES; ++j) {
            uint8_t byte = 0;
            for (int plane = 0; plane < 4; ++plane) {
                byte |= (random() % 3u) << (2 * plane);
            }
            block[j] = byte;
        }
        set_scale(block, 0.015625f + static_cast<float>(random() >> 8) / 16777216.0f);
    }
    return input;
}

std::array<double, TQ2_ROWS> reference_matvec(const Input &input) {
    std::array<double, TQ2_ROWS> result{};
    for (int row = 0; row < TQ2_ROWS; ++row) {
        double sum = 0.0;
        for (int b = 0; b < TQ2_BLOCKS_PER_ROW; ++b) {
            const uint8_t *block = input.weights.data() + (row * TQ2_BLOCKS_PER_ROW + b) * TQ2_BLOCK_BYTES;
            const double scale = read_scale(block);
            const float *x = input.activation.data() + b * TQ2_BLOCK_ELEMENTS;
            // Walk the logical activation order, unpacking one 32-element plane at a time.
            for (int half = 0; half < 2; ++half) {
                for (int plane = 0; plane < 4; ++plane) {
                    for (int column = 0; column < 32; ++column) {
                        const int code = (block[32 * half + column] >> (2 * plane)) & 3;
                        sum += (code - 1) * scale * *x++;
                    }
                }
            }
        }
        result[row] = sum;
    }
    return result;
}

// Uses the storage contract directly, independently of Metal coefficients.
double decoded_weight(const Input &input, size_t row, size_t column) {
    const size_t block_index = column / TQ2_BLOCK_ELEMENTS;
    const size_t position = column % TQ2_BLOCK_ELEMENTS;
    const size_t half = position / 128;
    const size_t plane = (position % 128) / 32;
    const size_t lane = position % 32;
    const uint8_t *block = input.weights.data() + (row * TQ2_BLOCKS_PER_ROW + block_index) * TQ2_BLOCK_BYTES;
    const int code = (block[32 * half + lane] >> (2 * plane)) & 3;
    return (code - 1) * read_scale(block);
}
