#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>

using namespace std;
using namespace chrono;

const int N = 2048;

// Генерация случайной матрицы
void generateMatrix(vector<double>& matrix) {

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<> dis(0.0, 1.0);

    for (double& x : matrix) {
        x = dis(gen);
    }
}

// 1 вариант — обычное умножение
void classicMultiply(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C
) {

    for (int i = 0; i < N; i++) {

        for (int j = 0; j < N; j++) {

            double sum = 0.0;

            for (int k = 0; k < N; k++) {

                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

// 3 вариант — оптимизированное умножение
void optimizedMultiply(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C
) {

    fill(C.begin(), C.end(), 0.0);

    for (int i = 0; i < N; i++) {

        double* cRow = &C[i * N];

        for (int k = 0; k < N; k++) {

            double r = A[i * N + k];

            const double* bRow = &B[k * N];

            for (int j = 0; j < N; j++) {

                cRow[j] += r * bRow[j];
            }
        }
    }
}

// Подсчет производительности
double calculatePerformance(double seconds) {

    double operations = 2.0 * N * N * N;

    return operations / seconds / 1e6;
}

int main() {

    vector<double> A(N * N);
    vector<double> B(N * N);

    vector<double> C1(N * N);
    vector<double> C2(N * N);
    vector<double> C3(N * N);

    cout << "Generating matrices..." << endl;

    generateMatrix(A);
    generateMatrix(B);

    // ==========================
    // 1 ВАРИАНТ
    // ==========================

    cout << endl;
    cout << "Classic multiplication..." << endl;

    auto start1 = high_resolution_clock::now();

    classicMultiply(A, B, C1);

    auto end1 = high_resolution_clock::now();

    double time1 =
        duration<double>(end1 - start1).count();

    double perf1 = calculatePerformance(time1);

    cout << "Time: " << time1 << " seconds" << endl;
    cout << "Performance: " << perf1 << " MFlops" << endl;

    // ==========================
    // 2 ВАРИАНТ BLAS
    // ==========================

    cout << endl;
    cout << "BLAS multiplication..." << endl;

    auto start2 = high_resolution_clock::now();

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        N,
        N,
        N,
        1.0,
        A.data(),
        N,
        B.data(),
        N,
        0.0,
        C2.data(),
        N
    );

    auto end2 = high_resolution_clock::now();

    double time2 =
        duration<double>(end2 - start2).count();

    double perf2 = calculatePerformance(time2);

    cout << "Time: " << time2 << " seconds" << endl;
    cout << "Performance: " << perf2 << " MFlops" << endl;

    // ==========================
    // 3 ВАРИАНТ
    // ==========================

    cout << endl;
    cout << "Optimized multiplication..." << endl;

    auto start3 = high_resolution_clock::now();

    optimizedMultiply(A, B, C3);

    auto end3 = high_resolution_clock::now();

    double time3 =
        duration<double>(end3 - start3).count();

    double perf3 = calculatePerformance(time3);

    cout << "Time: " << time3 << " seconds" << endl;
    cout << "Performance: " << perf3 << " MFlops" << endl;

    // ==========================
    // СРАВНЕНИЕ
    // ==========================

    double percent = perf3 / perf2 * 100.0;

    cout << endl;

    cout << "Optimized algorithm performance: "
         << percent
         << "% of BLAS performance"
         << endl;

    if (percent >= 30.0) {

        cout << "Requirement completed!" << endl;
    }
    else {

        cout << "Requirement NOT completed!" << endl;
    }

    return 0;
}