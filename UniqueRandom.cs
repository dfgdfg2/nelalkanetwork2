using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public class UniqueRandom
    {
        private Random _random;
        private HashSet<int> _generatedNumbers;
        private int _min;
        private int _max;

        public UniqueRandom(int min, int max)
        {
            _random = new Random();
            _generatedNumbers = new HashSet<int>();
            _min = min;
            _max = max;
        }

        public int GetUniqueRandomNumber()
        {
            if (_generatedNumbers.Count >= (_max - _min + 1))
            {
                throw new InvalidOperationException("All unique numbers have been generated.");
            }

            int number;
            do
            {
                number = _random.Next(_min, _max + 1);
            } while (_generatedNumbers.Contains(number));

            _generatedNumbers.Add(number);
            return number;
        }

        public void Reset()
        {
            _generatedNumbers.Clear();
        }
    }
}
