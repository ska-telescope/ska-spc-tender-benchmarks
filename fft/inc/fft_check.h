#include <complex>
#include <span>

#include "fft_configuration.h"

#define VALUE(string) #string
#define TO_LITERAL(string) VALUE(string)

namespace fft_benchmark
{
    template <typename real>
    auto compute_fft_error(const configuration &configuration, const std::span<std::complex<real>> &in,
                           const std::span<std::complex<real>> &out)
    {
        using complex = std::complex<real>;
#ifdef PYTHON_EXECUTABLE
        const std::string python_input_path = std::tmpnam(nullptr);
        const std::string python_output_path = std::tmpnam(nullptr);
        const std::string python_script_path = std::tmpnam(nullptr);

        const size_t sig_size = configuration.nx * configuration.ny;

        std::ofstream python_script_file{python_script_path};
        python_script_file << "import numpy as np\n";
        python_script_file << "data = np.fromfile('" << python_input_path << "', dtype=np."
                           << (configuration.ftype == float_type::single_precision ? "float32" : "float64")
                           << ", count=-1)\n";
        python_script_file << "cpx_data = data.view(dtype=np."
                           << (configuration.ftype == float_type::single_precision ? "complex64" : "complex128")
                           << ")\n";
        python_script_file << "cpx_data = np.reshape(cpx_data, (" << configuration.ny << "," << configuration.nx
                           << "))\n";
        python_script_file << "cpx_fft_data = np.fft.fft2(cpx_data, norm='forward')\n";
        python_script_file << "cpx_fft_data.tofile('" << python_output_path << "')\n";
        python_script_file.flush();

        std::ofstream python_input_file{python_input_path, std::ios::binary};
        python_input_file.write(reinterpret_cast<const char *>(in.data()), sig_size * sizeof(in[0]));
        python_input_file.close();

        std::stringstream python_command;
        python_command << TO_LITERAL(PYTHON_EXECUTABLE) << " " << python_script_path;
        std::system(python_command.str().c_str());

        std::vector<std::complex<double>> python_fft(sig_size);
        std::ifstream python_output_file{python_output_path, std::ios::binary};
        python_output_file.read(reinterpret_cast<char *>(python_fft.data()), python_fft.size() * sizeof(python_fft[0]));

        real max_error = std::abs(out[0] - complex(python_fft[0]));
        for (size_t i = 1; i < python_fft.size(); ++i)
        {
            const complex val = out[i];
            const complex py_val = complex(python_fft[i]);
            const real err = std::abs(val - py_val);
            const real abs_ref = std::abs(val);
            const real abs_py = real(std::abs(py_val));
            const real max_abs = std::max(abs_ref, abs_py);
            max_error = std::max(max_error, err) / std::max(real(1.), max_abs);
        }
        return max_error;
#else
        return std::numeric_limits<real>::quiet_NaN();
#endif
    }
} // namespace fft_benchmark