## Описание образца

Этот пример содержит пример умножения матриц с помощью CUBLAS. Реализация matmul повторно использует уже выделенные буферы на GPU, если размеры матриц совпадают. Он был разработан таким образом, чтобы сделать тесты производительности более «практичными», поскольку реальное приложение не выделяет память на GPU каждый раз при получении данных.

CUBLAS matmul сравнивается с другими реализациями:

- фиктивная реализация процессора (см. wrapper.pyx);
- реализация процессора с оптимизированным использованием памяти (см. wrapper.pyx). Столбцы матрицы B кэшируются для ускорения доступа к соответствующим элементам. Эта небольшая модификация приводит к увеличению производительности в 3 раза. Эта реализация не указана в таблице ниже. Запустите test.py, чтобы увидеть результаты.
- numpy.dot

## Performance research

The time measurements presented in the table below were averaged across 100. Only square matrices were used
during test. The time is measured in milliseconds.

| Mat size | CUBLAS | CPU | np.dot | CPU/CUBLAS | Numpy/CUBLAS |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.220 | 1.973 | __0.082__ | 8.968 | 0.372 |
| 256 | 0.307 | 15.41 | __0.280__ | 50.19 | 0.912 |
| 512 | __0.997__ | 274.6 | 1.480 | 275.4 | 1.484 |
| 1024 | __3.393__ | 5753 | 9.998 | 1695 | 2.947 |
| 2048 | __13.80__ | - | 75.47 | - | 5.469 |

The results suggest that for most cases Numpy will be a "go to" choice. CUDA should be used only in cases when matrices
larger 512 are used and being multiplied frequent enough to affect performance.

