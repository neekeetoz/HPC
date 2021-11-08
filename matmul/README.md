# Работа первая

Ход эксперименты находится в ноутбуке GPU_vs_CPU_exp.ipynb

Описание работы алгоритма на GPU
-------------------------
Умножение двух матриц используя несколько блоков и общую память. 
Каждому потоку блока присваивается тайл результирующей матрицы, который отвечает за расчет эелементов в этом тайле.
Каждой поток в блоке вычисляет один элемент в тайле.

Матрицы блоков вычисляют умножение двух блоков A и B. После этого сумма столбцов и строки сохраняется в результате для матрицы С.
В разделяемой памяти накапливается результат умножения каждым потоком.

Результаты
-------------------------

|Matrix size|CPU time, s|GPU time, s|Speedup|
|-----------|--------|--------|-------|
|128|2.3682| 0.0107|2.3575|
| 256|18.1763|0.0083| 18.1680|
| 512|153.1813|0.0127| 153.1685|
|1024|1268.3495| 0.0469| 1268.3025|


![Alt text](draw1.png?raw=true "Title")
![Alt text](draw2.png?raw=true "Title")

Выводы
-------------------------
Чем больше размер матриц, тем выше прирост скорости GPU. Это потому, что графическому процессору требуется некоторая предварительная обработка для реализации копий памяти и подготовки памяти.