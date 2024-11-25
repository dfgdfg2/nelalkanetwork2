using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public class Topology
    {
        public int InputCount { get; private set; }
        public int OutputCount { get; private set; }
        public List<int> HiddenCounts { get; private set; }
        public int Count { get { return HiddenCounts.Count + 2; } }
        public double LearningRate { get; private set; }
        public List<int> CollectionCounts { get; private set; }
        public int Epoch { get; private set; }
        public Topology(int inputs, int outputs, double learningRate, List<int> hiddens, int epoch)
        {
            Epoch = epoch;
            InputCount = inputs;
            OutputCount = outputs;
            LearningRate = learningRate;
            HiddenCounts = hiddens;
            CollectionCounts = new List<int>();
            CollectionCounts.Add(InputCount);
            for(int i = 0; i < HiddenCounts.Count; i++)
            {
                CollectionCounts.Add(HiddenCounts[i]);
            }
            CollectionCounts.Add(OutputCount);
        }
    }
}
