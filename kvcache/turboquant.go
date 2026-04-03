package kvcache

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/ml"
)

type TurboQuantState struct {
	mu        sync.Mutex
	enabled   bool
	headDim   int
	nKVHeads  int
	nLayers   int
	signs     []float32
	totalRows int64
}

func NewTurboQuantState(headDim, nKVHeads, nLayers int) *TurboQuantState {
	tq := &TurboQuantState{
		enabled:  true,
		headDim:  headDim,
		nKVHeads: nKVHeads,
		nLayers:  nLayers,
	}
	rng := uint64(42)
	tq.signs = make([]float32, headDim)
	for i := range tq.signs {
		rng ^= rng << 13
		rng ^= rng >> 7
		rng ^= rng << 17
		if rng&1 == 0 {
			tq.signs[i] = 1.0
		} else {
			tq.signs[i] = -1.0
		}
	}
	fmt.Fprintf(os.Stderr, "turboquant: initialized Go Phase 1 (head_dim=%d, nKVHeads=%d, nLayers=%d)\n",
		headDim, nKVHeads, nLayers)
	return tq
}

func f16ToF32(bits uint16) float32 {
	sign := uint32(bits>>15) & 1
	exp := uint32(bits>>10) & 0x1f
	frac := uint32(bits) & 0x3ff
	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3ff
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (frac << 13))
	}
	if exp == 31 {
		if frac == 0 {
			return math.Float32frombits((sign << 31) | 0x7f800000)
		}
		return math.Float32frombits((sign << 31) | 0x7f800000 | (frac << 13))
	}
	return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (frac << 13))
}

func f32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127
	frac := bits & 0x7fffff
	if exp > 15 {
		return sign | 0x7c00
	}
	if exp < -14 {
		return sign
	}
	return sign | uint16(exp+15)<<10 | uint16(frac>>13)
}

func (tq *TurboQuantState) quantizeDequantRow(data []float32) {
	d := tq.headDim
	if len(data) != d {
		return
	}
	for i := 0; i < d; i++ {
		data[i] *= tq.signs[i]
	}
	var sum float64
	for i := 0; i < d; i++ {
		sum += float64(data[i])
	}
	centroid := float32(sum / float64(d))
	var maxAbs float32
	for i := 0; i < d; i++ {
		data[i] -= centroid
		a := data[i]
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs < 1e-10 {
		for i := 0; i < d; i++ {
			data[i] = centroid * tq.signs[i]
		}
		return
	}
	scale := maxAbs / 3.5
	invScale := 3.5 / float64(maxAbs)
	for i := 0; i < d; i++ {
		q := math.Round(float64(data[i]) * invScale)
		if q < -3.5 {
			q = -3.5
		}
		if q > 3.5 {
			q = 3.5
		}
		data[i] = float32(q)*scale + centroid
	}
	for i := 0; i < d; i++ {
		data[i] *= tq.signs[i]
	}
}

// PostProcessKVTensor reads raw F16 bytes from the KV tensor via BackendGet,
// converts to F32, applies quantize+dequant, converts back to F16, and writes back.
func (tq *TurboQuantState) PostProcessKVTensor(tensor ml.Tensor, locations []int32) {
	if !tq.enabled || tensor == nil || len(locations) == 0 {
		return
	}

	tq.mu.Lock()
	defer tq.mu.Unlock()

	// BackendGet reads ggml_nbytes into a float32 array
	// For F16: nbytes = nelements * 2, so only first half of float32 array has data
	f32buf := tensor.BackendGet()
	if len(f32buf) == 0 {
		return
	}

	// Reinterpret the float32 buffer as raw bytes
	// BackendGet allocated nelements float32s = nelements*4 bytes
	// But ggml_backend_tensor_get only wrote nelements*2 bytes (F16)
	nElements := len(f32buf)
	rawBytes := unsafe.Slice((*byte)(unsafe.Pointer(&f32buf[0])), nElements*4)
	nbytes := nElements * 2 // F16: 2 bytes per element

	stride := tq.headDim * tq.nKVHeads
	rowBytes := stride * 2 // F16
	numCells := nbytes / rowBytes

	row := make([]float32, stride)

	for _, loc := range locations {
		if int(loc) >= numCells || loc < 0 {
			continue
		}
		rowStart := int(loc) * rowBytes

		// F16 -> F32
		for i := 0; i < stride; i++ {
			off := rowStart + i*2
			bits := binary.LittleEndian.Uint16(rawBytes[off : off+2])
			row[i] = f16ToF32(bits)
		}

		// Quantize+dequant per head
		for h := 0; h < tq.nKVHeads; h++ {
			offset := h * tq.headDim
			tq.quantizeDequantRow(row[offset : offset+tq.headDim])
		}

		// F32 -> F16
		for i := 0; i < stride; i++ {
			off := rowStart + i*2
			binary.LittleEndian.PutUint16(rawBytes[off:off+2], f32ToF16(row[i]))
		}
		tq.totalRows++
	}

	// Write back raw bytes via FromBytes
	// We need exactly nbytes bytes
	tensor.FromBytes(rawBytes[:nbytes])
}

func (tq *TurboQuantState) Enabled() bool {
	return tq != nil && tq.enabled
}
