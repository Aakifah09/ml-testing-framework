"""Subject programs used by the test-generation and test-oracle modules.

Keep these free of decorators and nested functions: the mutation engine and
the coverage tracer analyse each function's own source and code object.
"""


def triangle(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return "invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "invalid"
    if a == b and b == c:
        return "equilateral"
    if a == b or b == c or a == c:
        return "isosceles"
    return "scalene"


def days_in_month(year, month):
    if month < 1 or month > 12:
        return -1
    if month == 2:
        if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
            return 29
        return 28
    if month in (4, 6, 9, 11):
        return 30
    return 31


def gcd(a, b):
    a, b = abs(a), abs(b)
    while b != 0:
        a, b = b, a % b
    return a


def insertion_sort(xs):
    result = list(xs)
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result
