using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ConsoleApp1
{
    public class Neuron
    {
        public List<double> Weights { get; private set; }
        public double Delta { get; set; }

        public NeuronType neuronType { get; private set; }
        public double Output { get; set; }
        public int WeightsCount { get; private set; }
        public int NumberOfLayer{ get; private set; }
        public int NumberLayer { get; private set; }
        public Random rnd { get; private set; }
        [JsonConstructor]
        public Neuron(NeuronType neuronType, int weightsCount, int numberLayer, int numberOfLayer, Random rnd = null)
        {
            if (rnd != null)
            {
                this.rnd = rnd;
            }
            this.neuronType = neuronType;
            Weights = new List<double>();
            NumberOfLayer = numberOfLayer;
            NumberLayer = numberLayer;
            InitializeWeights(weightsCount, this.rnd);
        }

        private void InitializeWeights(int weightsCount, Random random = null)
        {
            if (neuronType == NeuronType.Input)
            {
                Weights.Add(1);
                WeightsCount = 1;
            }
            else
            {
                for (int i = 0; i < weightsCount; i++)
                {
                    Weights.Add(random.NextDouble());
                    WeightsCount += 1;
                }
            }
        }
    }
}
