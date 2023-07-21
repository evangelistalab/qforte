#include "tensor.h"


// May need an analog these eventually
// #include "../util/string.hpp"
// #include <lightspeed/math.hpp>

#include <iostream>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <cstring>

// namespace lightspeed { 

size_t Tensor::total_memory__ = 0;

/// Constructor
Tensor::Tensor(
    const std::vector<size_t>& shape,
    const std::string& name
    ) :
    shape_(shape),
    name_(name)
{
    strides_.resize(shape_.size());
    size_ = 1L;
    for (int i = shape_.size() - 1; i >= 0; i--) {
        strides_[i] = size_;
        size_ *= shape_[i];
    }  
    data_.resize(size_,0.0);

    // Ed's special memory thing
    total_memory__ += data_.size() * sizeof(std::complex<double>);
}

/// Destructor
Tensor::~Tensor()
{
    // Ed's special memory thing
    total_memory__ -= data_.size() * sizeof(std::complex<double>);
}

/// Set a particular element of tis Tensor, specified by idxs
// TODO(Nick) usde strides_
void Tensor::set(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    ndim_error(idxs.size());

    if( idxs.size() == 1 ) {
        data_[idxs[0]] = val;
    } else if (idxs.size() == 2) {
        data_[shape()[1]*idxs[0] + idxs[1]] = val;
    } else {
        for (int i = 0; i < ndim(); i++) {
            if (idxs[i] < 0 || idxs[i] >= shape()[i]) {
                std::cerr << "Index out of bounds for dimension " << i << std::endl;
            }
        }      
        size_t vidx = 0;
        size_t stride = 1;
        
        for (int i = ndim() - 1; i >= 0; i--) {
            vidx += idxs[i] * stride;
            stride *= shape()[i];
        }
        data_[vidx] = val;
    } 
}

/// Get the vector index for this tensor based on the tensor index
size_t Tensor::tidx_to_vidx(const std::vector<size_t>& tidx) const
{   
    size_t vidx = 0;
    for (int i = ndim() - 1; i >= 0; i--) {
        vidx += tidx[i] * strides_[i];
    }
    return vidx;
}

size_t Tensor::tidx_to_trans_vidx(const std::vector<size_t>& tidx, const std::vector<size_t>& axes) const
{   
    size_t vidx = 0;
    for (int i = ndim() - 1; i >= 0; i--) {
        vidx += tidx[i] * strides_[axes[i]];
    }
    return vidx;
}

/// Get the tensor index for this tensor based on the vector index
std::vector<size_t> Tensor::vidx_to_tidx(size_t vidx) const
{
    std::vector<size_t> tidx(ndim());
    size_t vidx_tmp = vidx;

    for (int i = ndim() - 1; i >= 0; i--) {
        tidx[i] = vidx_tmp % shape()[i];
        vidx_tmp /= shape()[i];
    }
    return tidx;
}

/// Get a particular element of tis Tensor, specified by idxs
// TODO(Nick/Tyler) use strides_
std::complex<double> Tensor::get(
    const std::vector<size_t>& idxs
    ) const
{
    ndim_error(idxs.size());

    if( idxs.size() == 1 ) {
        return data_[idxs[0]];
    } else if (idxs.size() == 2) {
        return data_[shape()[1]*idxs[0] + idxs[1]];
    } else {
        for (int i = 0; i < ndim(); i++) {
            if (idxs[i] < 0 || idxs[i] >= shape()[i]) {
                std::cerr << "Index out of bounds for dimension " << i << std::endl;
            }
        }
        
        size_t vidx = 0;
        size_t stride = 1;
        
        for (int i = ndim() - 1; i >= 0; i--) {
            vidx += idxs[i] * stride;
            stride *= shape()[i];
        }

        return data_[vidx];
    }
}

void Tensor::ndim_error(size_t ndims) const
{
    if (!(ndim() == ndims)) {
        std::stringstream ss;
        ss << "Tensor should be " << ndims << " ndim, but is " << ndim() << " ndim.";
        throw std::runtime_error(ss.str());
    }
}

void Tensor::shape_error(const std::vector<size_t>& shape) const
{
    ndim_error(shape.size());
    for (size_t i = 0; i < ndim(); i++) {
        if (shape_[i] != shape[i]) {
            std::stringstream ss;
            ss << "Tensor should be (";
            for (size_t j = 0; j < ndim(); j++) {
                ss << shape[j];
                if (j < ndim() - 1) {
                    ss << ",";
                }
            }
            ss << ") shape, but is (";
            for (size_t j = 0; j < ndim(); j++) {
                ss << shape_[j];
                if (j < ndim() - 1) {
                    ss << ",";
                }
            }
            ss << ") shape.";
            throw std::runtime_error(ss.str());
        }
    }
}

void Tensor::square_error() const 
{
    ndim_error(2);
    if (shape_[0] != shape_[1]) {
        std::stringstream ss;
        ss << "Tensor should be square, but is ";
        ss << "(" << shape_[0] << "," << shape_[1] << ") shape.";
        throw std::runtime_error(ss.str());
    }
}

std::shared_ptr<Tensor> Tensor::clone()
{
    return std::shared_ptr<Tensor>(new Tensor(*this)); 
}

void Tensor::zero()
{
    memset(data_.data(),'\0',sizeof(std::complex<double>)*size_);
}

void Tensor::identity()
{
    square_error();
    zero();
    for (size_t i = 0; i < shape_[0]; i++) {
        data_[i * shape_[1] + i] = 1.0;
    }
}

void Tensor::symmetrize()
{
    square_error();
    for (size_t i = 0; i < shape_[0]; i++) {
        for (size_t j = 0; j < shape_[0]; j++) {
            data_[i * shape_[1] + j] =
            data_[j * shape_[1] + i] = 0.5 * (
            data_[i * shape_[1] + j] +
            data_[j * shape_[1] + i]);
        }
    }
}

void Tensor::antisymmetrize()
{
    square_error();
    for (size_t i = 0; i < shape_[0]; i++) {
        for (size_t j = 0; j < shape_[0]; j++) {
            std::complex<double> val = 0.5 * (
            data_[i * shape_[1] + j] -
            data_[j * shape_[1] + i]);
            data_[i * shape_[1] + j] = val;
            data_[j * shape_[1] + i] = - val;
        }
    }
}

// TODO(NICK:) reimplement Scal
void Tensor::scale(std::complex<double> a)
{
    // C_DSCAL(size_,a,data_.data(),1);
    for(size_t i = 0; i < size_; i++){
        data_[i] *= a;
    }
}

void Tensor::copy(
    const std::shared_ptr<Tensor>& other
    )
{
    shape_error(other->shape());
    
    std::memcpy(data_.data(), other->data().data(), sizeof(std::complex<double>)*size_);
}

void Tensor::add(const Tensor& other) 
{
    shape_error(other.shape());
    for(size_t i = 0; i < size_; i++){
        data_[i] += other.read_data()[i];
    }

    // std::transform(
    //     data_.begin(), 
    //     data_.end(), 
    //     other.read_data().begin(),
    //     data_.begin(), 
    //     add_c<double>
    //     );
}

void Tensor::subtract(const Tensor& other){

    shape_error(other.shape());
    for (size_t i = 0; i < size_; i++){
        data_[i] -= other.read_data()[i];
    }
}

double Tensor::norm(){

    double result = 0;

    for (int i = 0; i < size_; i++){

        result += std::real(data_[i]) * std::real(data_[i]) + std::imag(data_[i]) * std::imag(data_[i]);

    }

    result = std::sqrt(result);

    return result;

}

// void Tensor::axpby(
//     const std::shared_ptr<Tensor>& other,
//     std::complex<double> a,
//     std::complex<double> b
//     )
// {
//     shape_error(other->shape());
    
//     C_DSCAL(size_,b,data_.data(),1);
//     C_DAXPY(size_,a,other->data().data(),1,data_.data(),1); 
// }

// std::complex<double> Tensor::vector_dot(
//     const std::shared_ptr<Tensor>& other
//     ) const
// {
//     shape_error(other->shape());

//     return C_DDOT(size_, const_cast<std::complex<double>*>(data_.data()), 1, other->data().data(), 1);
// }

// NOTE(Nick) we maywant to return sharred pointer to a tensor instead...
// std::shared_pointer<Tensor> Tensor::transpose() const
Tensor Tensor::transpose() const
{
    ndim_error(2);
    // std::shared_ptr<Tensor> T(new Tensor({shape_[1], shape_[0]}));
    Tensor T({shape_[1], shape_[0]});
    std::complex<double>* Tp = T.data().data();
    const std::complex<double>* Ap = data_.data();
    for (size_t ind1 = 0; ind1 < shape_[0]; ind1++) {
        for (size_t ind2 = 0; ind2 < shape_[1]; ind2++) {
            Tp[ind2 * shape_[0] + ind1] = Ap[ind1 * shape_[1] + ind2];
        }
    }
    return T;
}

// NOTE(Nick) we maywant to return sharred pointer to a tensor instead...
Tensor Tensor::general_transpose(const std::vector<size_t>& axes) const 
{
    if (axes.size() != ndim()) {
        throw std::invalid_argument("Invalid axes permutation");
    }

    std::vector<size_t> transposed_shape(ndim());
    for (size_t i = 0; i < ndim(); ++i) {
        transposed_shape[i] = shape_[axes[i]];
    }

    // std::shared_ptr<Tensor> transposed_tensor(new Tensor(transposed_shape));
    Tensor transposed_tensor(transposed_shape);

    std::complex<double>* transposed_data = transposed_tensor.data().data();
    // const std::complex<double>* original_data = data_.data();

    // This works but probably can be made more efficient.
    // Fix if it turns out to be a bottleneck
    for (size_t i = 0; i < size_; i++){
        std::vector<size_t> tidx_trans = vidx_to_tidx(i);
        size_t t_vidx = transposed_tensor.tidx_to_trans_vidx(tidx_trans, axes);
        transposed_data[t_vidx] = data_[i];
    }

    return transposed_tensor;  
}

// TODO(Tyler?): Column printing is a little clunky for complex
// need to fix
std::string Tensor::str(
    bool print_data, 
    bool print_complex, 
    int maxcols,
    const std::string& data_format,
    const std::string& header_format
    ) const
{
    std::string str = "";
    str += std::printf( "Tensor: %s\n", name_.c_str());
    str += std::printf( "  Ndim  = %zu\n", ndim());
    str += std::printf( "  Size  = %zu\n", size());
    str += std::printf( "  Shape = (");
    for (size_t dim = 0; dim < ndim(); dim++) {
        str += std::printf( "%zu", shape_[dim]);
        if (dim < ndim() - 1) {
            str += std::printf( ",");
        }
    }
    str += std::printf(")\n");

    if (print_data) {

        std::string data_format2 = data_format;

        if(print_complex){ data_format2 = "%f%+fi";}

        str += std::printf("\n");
            
        std::string order0str1 = "  " + data_format2 + "\n";
        std::string order1str1 = "  %5zu " + data_format2 + "\n";
        std::string order2str1 = " " + header_format;
        std::string order2str2 = " " + data_format2;

        int order = ndim();
        size_t nelem = size();

        size_t page_size = 1L;
        size_t rows = 1;
        size_t cols = 1;
        if (order >= 1) {
            page_size *= shape_[order - 1];
            rows = shape_[order - 1];
        }
        if (order >= 2) {
            page_size *= shape_[order - 2];
            rows = shape_[order - 2];
            cols = shape_[order - 1];
        }

        str += std::printf( "  Data:\n\n");

        if (nelem > 0){
            size_t pages = nelem / page_size;
            for (size_t page = 0L; page < pages; page++) {

                if (order > 2) {
                    str += std::printf( "  Page (");
                    size_t num = page;
                    size_t den = pages;
                    size_t val;
                    for (int k = 0; k < order - 2; k++) {
                        den /= shape_[k];
                        val = num / den;
                        num -= val * den;
                        str += std::printf("%zu,",val);
                    }
                    str += std::printf( "*,*):\n\n");
                }

                const std::complex<double>* vp = data_.data() + page * page_size;
                if (order == 0) {
                    str += std::printf( order0str1.c_str(), *(vp));
                } else if(order == 1) {
                    for (size_t i=0; i<page_size; ++i) {
                        str += std::printf( order1str1.c_str(), i, *(vp + i));
                    }
                } else {
                    for (size_t j = 0; j < cols; j += maxcols) {
                        size_t ncols = (j + maxcols >= cols ? cols - j : maxcols);
                
                        // Column Header
                        str += std::printf("  %5s", "");
                        for (size_t jj = j; jj < j+ncols; jj++) {
                            str += std::printf(order2str1.c_str(), jj);
                        }
                        str += std::printf("\n");

                        // Data
                        for (size_t i = 0; i < rows; i++) {
                            str += std::printf("  %5zu", i);
                            for (size_t jj = j; jj < j+ncols; jj++) {
                                str += std::printf(order2str2.c_str(), *(vp + i * cols + jj));
                            }
                            str += std::printf("\n");
                        }

                        // Block separator
                        if (page < pages - 1 || j + maxcols < cols - 1) str += std::printf("\n");
                    }
                }
            }
        }
    }
    return str;
}


// py::array_t<double> array
// std::vector<std::complex<double>>

void Tensor::fill_from_nparray(std::vector<std::complex<double>> arr, std::vector<size_t> shape){

    if (shape_ != shape){
        throw std::runtime_error("The Shapes are not the same.");
    }

    std::memcpy(data_.data(), arr.data(), sizeof(std::complex<double>)*size_);

}

// TODO(Nick): Re-Implement
// void Tensor::print() const
// {
//     std::cout << string() << std::endl;
// }

// void Tensor::print(const std::string& name)
// {
//     std::string bak_name = name_;
//     set_name(name);
//     print();
//     set_name(bak_name);
// }

// } // namespace lightspeed