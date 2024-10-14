using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public class Layer
    {
        public List<Neuron> Neurons { get; private set; } 
        public int Count { get { return Neurons.Count; } private set { } }
        public NeuronType LayerType { get; private set; }

        public Layer(List<Neuron> neurons) 
        {
            this.Neurons = neurons;
            LayerType = neurons[0].neuronType;
        }
        public Neuron this[int index]
        {
            get { return Neurons[index]; } 
            set { Neurons[index] = value; }
        }

        public List<double> GetSignals()
        {
            List<double> arr = new List<double>(Neurons.Count);
            for (int i = 0; i < Neurons.Count; i++)
            {
                double output = Neurons[i].Output;
                arr.Add(output);
            }
            return arr;
        }
    }
}
