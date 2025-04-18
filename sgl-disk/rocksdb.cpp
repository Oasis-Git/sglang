#include <torch/extension.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <limits>
#include <stdexcept>

inline rocksdb::Slice tensorToSlice(const torch::Tensor& t) {
    if (!t.is_contiguous()) {
        throw std::runtime_error("tensorToSlice: tensor must be contiguous");
    }

    const void* data_ptr = t.data_ptr();
    if (t.numel() && data_ptr == nullptr) {
        throw std::runtime_error("tensorToSlice: non-zero tensor with null data ptr");
    }

    const uint64_t numel = static_cast<uint64_t>(t.numel());
    const uint64_t esize = static_cast<uint64_t>(t.element_size());
    if (esize && numel > std::numeric_limits<uint64_t>::max() / esize) {
        throw std::runtime_error("tensorToSlice: size overflow");
    }

    size_t bytes = static_cast<size_t>(numel * esize);
    return rocksdb::Slice(reinterpret_cast<const char*>(data_ptr), bytes);
}

inline void sliceToTensor(const rocksdb::Slice& s, torch::Tensor& out) {
    if (!out.is_contiguous()) {
        throw std::runtime_error("sliceToTensor: output tensor must be contiguous");
    }

    const uint64_t numel = static_cast<uint64_t>(out.numel());
    const uint64_t esize = static_cast<uint64_t>(out.element_size());
    if (esize && numel > std::numeric_limits<uint64_t>::max() / esize) {
        throw std::runtime_error("sliceToTensor: size overflow");
    }

    size_t expected = static_cast<size_t>(numel * esize);
    if (expected != s.size()) {
        throw std::runtime_error("sliceToTensor: size mismatch");
    }

    if (expected) {
        std::memcpy(out.data_ptr(), s.data(), expected);
    }
}

class RocksDBStorageBackend {
private:
    rocksdb::DB* db;
    rocksdb::Options options;
    std::string db_path_;
    rocksdb::WriteOptions write_opts_;
    bool is_open;

public:
    RocksDBStorageBackend() : db(nullptr), is_open(false) {}
    RocksDBStorageBackend(const RocksDBStorageBackend&) = delete;
    RocksDBStorageBackend& operator=(const RocksDBStorageBackend&) = delete;

    RocksDBStorageBackend(RocksDBStorageBackend&& other) noexcept 
        : db(other.db),
          options(std::move(other.options)),
          db_path_(std::move(other.db_path_)),
          write_opts_(std::move(other.write_opts_)),
          is_open(other.is_open) {
        other.db = nullptr;
        other.is_open = false;
    }

    RocksDBStorageBackend& operator=(RocksDBStorageBackend&& other) noexcept {
        if (this != &other) {
            close();
            db = other.db;
            options = std::move(other.options);
            db_path_ = std::move(other.db_path_);
            write_opts_ = std::move(other.write_opts_);
            is_open = other.is_open;
            other.db = nullptr;
            other.is_open = false;
        }
        return *this;
    }

    ~RocksDBStorageBackend() { close(); }

    void init(const std::string& db_path, size_t blob_file_size = 256 * 1024 * 1024) {
        if (is_open) {
            throw std::runtime_error("Database already initialized. Call close() first.");
        }

        db_path_ = db_path;
        options.create_if_missing = true;
        options.IncreaseParallelism();
        options.OptimizeLevelStyleCompaction();

        // Configure blob storage
        options.enable_blob_files = true;
        options.min_blob_size = 1024 * 1024;
        options.blob_file_size = blob_file_size;
        options.enable_blob_garbage_collection = false;
        // options.blob_compression = rocksdb::kNoCompression;

        // Configure write options
        write_opts_.disableWAL = true;
        write_opts_.sync = false;
        // write_opts_.no_slowdown = true;

        rocksdb::Status status = rocksdb::DB::Open(options, db_path_, &db);
        if (!status.ok()) {
            throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
        }

        is_open = true;
        std::cout << "RocksDB initialized at: " << db_path_ 
                  << " with blob_file_size: " << options.blob_file_size << std::endl;
    }

    void storekv(const std::string& hash, const torch::Tensor& tensor) {
        if (!is_open) {
            throw std::runtime_error("Database not initialized.");
        }

        rocksdb::Slice key(hash);
        rocksdb::Slice value = tensorToSlice(tensor);
        rocksdb::Status status = db->Put(write_opts_, key, value);
        if (!status.ok()) {
            throw std::runtime_error("Failed to PUT key '" + hash + "': " + status.ToString());
        }
    }

    void storeBatch(const std::vector<std::pair<std::string, torch::Tensor>>& kvs) {
        if (!is_open) {
            throw std::runtime_error("Database not initialized.");
        }

        rocksdb::WriteBatch batch;
        for (const auto& [key, tensor] : kvs) {
            batch.Put(rocksdb::Slice(key), tensorToSlice(tensor));
        }
        rocksdb::Status status = db->Write(write_opts_, &batch);
        if (!status.ok()) {
            throw std::runtime_error("Failed to PUT batch: " + status.ToString());
        }
    }

    void retrievekv(const std::string& hash, torch::Tensor& kv) {
        if (!is_open) {
            throw std::runtime_error("Database not initialized.");
        }

        if (!kv.is_contiguous()) {
            throw std::runtime_error("Target tensor 'kv' must be contiguous.");
        }

        if (kv.numel() > 0 && kv.data_ptr() == nullptr) {
            throw std::runtime_error("Target tensor 'kv' has non-zero size but null data pointer.");
        }

        rocksdb::ReadOptions read_options;
        rocksdb::PinnableSlice value;
        rocksdb::Status status = db->Get(read_options, db->DefaultColumnFamily(), hash, &value);

        if (status.IsNotFound()) {
            throw std::runtime_error("Key not found: " + hash);
        } else if (!status.ok()) {
            throw std::runtime_error("Failed to GET key '" + hash + "': " + status.ToString());
        }

        try {
            sliceToTensor(value, kv);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed during deserialization for key '" + hash + "': " + e.what());
        }
    }

    void retrieveBatch(
        const std::vector<std::string>&   keys,
        std::vector<torch::Tensor>&       outs
    ) {
        if (!is_open) throw std::runtime_error("Database not initialized.");
        if (keys.size() != outs.size())
            throw std::runtime_error("retrieveBatch: keys and output tensors count mismatch");

        rocksdb::ReadOptions ro;

        std::vector<rocksdb::Slice> key_slices;
        key_slices.reserve(keys.size());
        for (const auto &k : keys) {
            key_slices.emplace_back(k);
        }

        std::vector<std::string> raw_values(keys.size());

        auto statuses = db->MultiGet(ro, key_slices, &raw_values);

        for (size_t i = 0; i < keys.size(); ++i) {
            const auto &st = statuses[i];
            if (st.IsNotFound()) {
            throw std::runtime_error("Key not found: " + keys[i]);
            } else if (!st.ok()) {
            throw std::runtime_error("MultiGet failed for " + keys[i] + ": " + st.ToString());
            }
            // sliceToTensor checks size match and memcpy
            rocksdb::Slice val_slice(raw_values[i]);
            sliceToTensor(val_slice, outs[i]);
        }
    }

        void remove(const std::string& hash) {
            if (!is_open) {
                throw std::runtime_error("Database not initialized.");
            }

            rocksdb::Status status = db->Delete(write_opts_, rocksdb::Slice(hash));
            if (!status.ok() && !status.IsNotFound()) {
                throw std::runtime_error("Failed to DELETE key '" + hash + "': " + status.ToString());
            }
        }

        void close() {
            if (is_open && db != nullptr) {
                rocksdb::Status s = db->Close();
                if (!s.ok()) {
                    std::cerr << "Warning: RocksDB Close() failed: " + s.ToString() << std::endl;
                }
                delete db;
                db = nullptr;
                is_open = false;
                std::cout << "RocksDB closed: " << db_path_ << std::endl;
            }
        }
    };

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<RocksDBStorageBackend>(m, "RocksDBStorageBackend")
        .def(py::init<>())
        .def("init", &RocksDBStorageBackend::init,
             py::arg("db_path"),
             py::arg("blob_file_size") = 256 * 1024 * 1024)
        .def("storekv", &RocksDBStorageBackend::storekv,
             py::arg("hash"),
             py::arg("tensor"))
        .def("storekv_batch",
             [](RocksDBStorageBackend& self, const py::iterable& kvs) {
                 std::vector<std::pair<std::string, torch::Tensor>> vec;
                 for (auto item : kvs) {
                     auto tup = item.cast<py::tuple>();
                     if (tup.size() != 2) {
                         throw std::runtime_error("Each element must be (key, tensor)");
                     }
                     vec.emplace_back(
                         tup[0].cast<std::string>(),
                         tup[1].cast<torch::Tensor>()
                     );
                 }
                 self.storeBatch(vec);
             },
             py::arg("kvs"),
             R"pbdoc(
                Atomically write a batch of (key, tensor) pairs
                using the WAL settings from init(...)
             )pbdoc")
        .def("retrievekv", &RocksDBStorageBackend::retrievekv,
             py::arg("hash"),
             py::arg("kv"))
        .def(
        "retrievekv_batch",
        [](RocksDBStorageBackend &self, const py::iterable &kvs) {
            std::vector<std::string>   keys;
            std::vector<torch::Tensor> outs;

            for (auto item : kvs) {
            auto tup = item.cast<py::tuple>();
            if (tup.size() != 2)
                throw std::runtime_error("Each element must be (key, tensor)");
            keys.push_back( tup[0].cast<std::string>() );
            auto t = tup[1].cast<torch::Tensor>();
            if (!t.is_contiguous())
                throw std::runtime_error("retrieve_batch: tensor must be contiguous");
            outs.push_back(t);
            }

            self.retrieveBatch(keys, outs);
            // no need to return; tensors are mutated in-place
        },
        py::arg("kvs"),
        R"pbdoc(
            Batch‑retrieve (key, tensor) pairs.
            Keys are looked up via MultiGet; values are copied into your pre‑allocated tensors.
        )pbdoc"
        )
        .def("remove", &RocksDBStorageBackend::remove,
             py::arg("hash"))
        .def("close", &RocksDBStorageBackend::close);
}