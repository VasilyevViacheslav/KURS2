#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <set>
#include <Eigen/Dense>


using namespace Eigen;
using namespace std;    

class Point {
public:
    Point() {}
    Point(const std::vector<double>& coords) : coords_(coords) {}

    std::size_t size() const { return coords_.size(); }
    double& operator[](std::size_t i) { return coords_[i]; }
    const double& operator[](std::size_t i) const { return coords_[i]; }

    std::vector<double> homogeneous() const {
        std::vector<double> hcoords(coords_.begin(), coords_.end());
        hcoords.push_back(1.0);
        return hcoords;
    }

private:
    std::vector<double> coords_;
};

std::vector<std::vector<double>> random_unit_vectors(int d, int n) {
    // —оздаем генератор случайных чисел
    std::mt19937 gen{ std::random_device{}() };
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // —оздаем матрицу векторов размера n x d
    std::vector<std::vector<double>> vectors(n, std::vector<double>(d));

    // «аполн€ем матрицу случайными числами и нормализуем каждый вектор
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            vectors[i][j] = dist(gen);
        }

        double norm = 0.0;
        for (int j = 0; j < d; ++j) {
            norm += vectors[i][j] * vectors[i][j];
        }
        norm = std::sqrt(norm);

        for (int j = 0; j < d; ++j) {
            vectors[i][j] /= norm;
        }
    }

    return vectors;
}
vector<double> find_furthest_point(vector<vector<double>> points, vector<vector<double>> vectors) {
    // Find the point in `points` that is furthest in each direction of `vectors`.
    auto n = points.size();
    auto d = points[0].size();
    auto m = vectors.size();
    vector<double> max_dots(m, -numeric_limits<double>::infinity());
    vector<int> indices(m, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double dot = 0.0;
            for (int k = 0; k < d; ++k) {
                dot += points[i][k] * vectors[j][k];
            }
            if (dot > max_dots[j]) {
                max_dots[j] = dot;
                indices[j] = i;
            }
        }
    }
    vector<double> result(m * d);
    for (int j = 0; j < m; ++j) {
        for (int k = 0; k < d; ++k) {
            result[j * d + k] = points[indices[j]][k];
        }
    }
    return result;
}
std::vector<std::vector<double>> random_unit_vectors(int d, int n) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<std::vector<double>> vectors(n, std::vector<double>(d, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            vectors[i][j] = distribution(generator);
        }
        double norm = std::sqrt(std::inner_product(vectors[i].begin(), vectors[i].end(), vectors[i].begin(), 0.0));
        std::transform(vectors[i].begin(), vectors[i].end(), vectors[i].begin(), [norm](double x) {return x / norm; });
    }
    return vectors;
}

// Find the point in `points` that is furthest in each direction of `vectors`.

bool locate_point_in_simplex(const Eigen::VectorXd& point, const Eigen::MatrixXd& simplex) {
    int d = simplex.cols() - 1;
    Eigen::MatrixXd A(d + 1, d + 1);
    A.block(0, 0, d, d) = simplex.transpose();
    A.col(d) = Eigen::VectorXd::Ones(d + 1);
    A.row(d).setZero();
    A(d, d) = 1;
    Eigen::VectorXd b(d + 1);
    b.head(d) = point;
    b(d) = 1;
    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
    return (x.array() >= 0).all();
}
int main() {
    int d = 3;
    int n = 4;
    auto vectors = random_unit_vectors(d, n);
    for (const auto& vec : vectors) {
        for (const auto& coord : vec) {
            std::cout << coord << " ";
        }
        std::cout << std::endl;
    }

    std::vector<std::vector<double>> points = { {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0} };
    auto furthest_point = find_furthest_point(points, vectors);
    for (const auto& coord : furthest_point) {
        std::cout << coord << " ";
    }
    std::cout << std::endl;

    return 0;
}