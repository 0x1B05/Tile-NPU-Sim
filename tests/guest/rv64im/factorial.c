// 10! = 3628800, exercising mul, a store/load round-trip, and a branch loop.
// Convention: main returns 0 on pass (reported as ebreak with a0 == 0).

static volatile unsigned long long result;

int main(void) {
  volatile unsigned long long n = 10;
  unsigned long long acc = 1;
  while (n != 0) {
    acc *= n;
    --n;
  }
  result = acc;
  return (acc == 3628800 && result == 3628800) ? 0 : 1;
}
