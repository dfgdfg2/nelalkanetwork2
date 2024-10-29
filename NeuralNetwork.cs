using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO;
using Newtonsoft.Json;


namespace ConsoleApp1
{
    public class NeuralNetwork
    {
        public List<Layer> neuronLayer { get; private set; }
        public Topology Topology { get; private set; }
        public ITestRegularization Reg { get; set; }
        public JsonSerializerOptions JsonSerializerOptions { get; set; } = new JsonSerializerOptions
        {
                IncludeFields = true,
                PropertyNameCaseInsensitive = true
        };
        public NeuralNetwork(Topology topology)
        {
            this.Topology = topology;
            neuronLayer = new List<Layer>();

            InitializeNeurons();
        }
        public double GetSumWeights()
        {
            double sumWeights = 0;
            for (int i = 0; i < neuronLayer.Count; i++)
            {
                sumWeights += neuronLayer[i].SumWeights();
            }
            return sumWeights;
        }

        public void WithdrawDates(string path)
        {
            List<Layer> dates = null;
            using (StreamReader sr = new StreamReader(path))
            {
                string json = sr.ReadToEnd();
                dates = JsonConvert.DeserializeObject<List<Layer>>(json);
            }
            for (int i = 0; i < dates.Count; i++)
            {
                Layer layer = dates[i];
                for (int j = 0; j < layer.Count; j++)
                {
                    Neuron neuron = layer[j];
                    for (int k = 0; k < neuron.WeightsCount; k++)
                    {
                        Console.Write(neuron.Weights[k] + "\u00A0\u00A0");
                    }
                }
                Console.WriteLine();
            }
        }
        public void GetDatesTo(string path)
        {
            using(FileStream fs = new FileStream(path, FileMode.OpenOrCreate))
            {
                System.Text.Json.JsonSerializer.Serialize(fs, neuronLayer);
            }
        }

        public double Learn(List<double> inputs, int[] expected, int epoch, AbGradientDescent descent = null, Normalization normalization = Normalization.None)
        {
            if (descent == null)
            {
                descent = new GradientDescent(this);
            }
            return descent.Backpropagation(inputs, expected, epoch);
        }

        public double Learn(double[][] dataset, int[] expected, int epoch, AbGradientDescent descent = null, Normalization normalization = Normalization.None)
        {
            if (descent == null)
            {
                descent = new GradientDescent(this);
            }
            if (normalization == Normalization.NormalizationZ)
            {
                NormalizationZ(dataset);
            }
            else if (normalization == Normalization.MinMax)
            {
                MinMax(dataset);
            }
            double error = descent.Backpropagation(dataset, expected, epoch);
            
            return error;
        }
        public double FeedForward(List<double> inputs)
        {
            return Reg.FeedForward(inputs, neuronLayer);
        }

        public double FeedForward(double[] inputs)
        {
            return Reg.FeedForward(inputs, neuronLayer);
        }

        private void NormalizationZ(double[][] inputs)
        { 
            int i = 0;
            int j = 0;
            int length1 = inputs[0].Length;
            int length2 = inputs.Length;
            double[] avg = new double[length1];
            double[] delta = new double[length1];
            for (i = 0; i < length1; i++)
            {
                double aver = 0;
                for (j = 0; j < length2; j++)
                {
                    aver += inputs[j][i];
                }
                aver /= length2;
                avg[i] = aver;
            }
            for (i = 0; i < length1; i++)
            {
                double d = 0;
                for (j = 0; j < length2; j++)
                {
                    d += Math.Pow(inputs[j][i] - avg[i], 2);
                }
                d /= length2;
                delta[i] = d;
            }
            for (i = 0; i < length1; i++)
            {
                for (j = 0; j < length2; j++)
                {
                    double currentN = inputs[j][i];
                    inputs[j][i] = (currentN - avg[i]) / delta[i];
                }
            }
        }

        private void MinMax(double[][] dataset)
        {
            int length1 = dataset[0].Length;
            int length2 = dataset.Length;
            double[] max = new double[length1];
            double[] min = new double[length1];
            for(int i = 0; i < length1; i++)
            {
                for (int j = 0; j < length2; j++)
                {
                    double Data = dataset[j][i];
                    if (Data > max[i])
                    {
                        max[i] = Data;
                    }
                    if (Data < min[i])
                    {
                        min[i] = Data;
                    }
                }
            }

            for (int i = 0; i < length1; i++)
            {
                for (int j = 0; j < length2; j++)
                {
                    dataset[j][i] = (dataset[j][i] - min[i]) / (max[i] - min[i]);
                }
            }
        }

        private void InitializeNeurons()
        {
            List<Neuron> currentLayer = new List<Neuron>(Topology.InputCount);
            int WeightsCount = 1;
            Random random = new Random();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                currentLayer.Add(new Neuron(NeuronType.Input, WeightsCount, 0, i));
            }
            neuronLayer.Add(new Layer(currentLayer));
            for (int i = 0; i < Topology.HiddenCounts.Count; i++)
            {
                WeightsCount = neuronLayer.Last().Count;
                currentLayer = new List<Neuron>(Topology.HiddenCounts[i]);
                for (int j = 0; j < Topology.HiddenCounts[i]; j++)
                {
                    currentLayer.Add(new Neuron(NeuronType.Hidden, WeightsCount, i + 1, j, random));
                }
                neuronLayer.Add(new Layer(currentLayer));
            }
            currentLayer = new List<Neuron>(Topology.OutputCount);
            WeightsCount = neuronLayer.Last().Count;
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                currentLayer.Add(new Neuron(NeuronType.Output, WeightsCount, Topology.Count - 1, i, random));
            }
            neuronLayer.Add(new Layer(currentLayer));
        }
    }
}
