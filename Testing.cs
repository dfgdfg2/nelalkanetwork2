using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    internal class Testing
    {
        static void Main(string[] args)
        {
            int[] outputs = new int[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            double[][] inputs =
            {
                new double[] { 0, 0, 0, 0 },
                new double[] { 0, 0, 0, 1 },
                new double[] { 0, 0, 1, 0 },
                new double[] { 0, 0, 1, 1 },
                new double[] { 0, 1, 0, 0 },
                new double[] { 0, 1, 0, 1 },
                new double[] { 0, 1, 1, 0 },
                new double[] { 0, 1, 1, 1 },
                new double[] { 1, 0, 0, 0 },
                new double[] { 1, 0, 0, 1 },
                new double[] { 1, 0, 1, 0 },
                new double[] { 1, 0, 1, 1 },
                new double[] { 1, 1, 0, 0 },
                new double[] { 1, 1, 0, 1 },
                new double[] { 1, 1, 1, 0 },
                new double[] { 1, 1, 1, 1 }
            };
            Topology top = new Topology(4, 1, 0.1, new List<int> { 2 });
            NeuralNetwork nn = new NeuralNetwork(top);
            double result = nn.Learn(inputs, outputs, 1000, new GradientDescent(nn, new Momentum(top)));

            Console.WriteLine($"avg: {result}");
            int length = inputs.GetLength(0);

            for (int i = 0; i < length; i++)
            {
                int expected = outputs[i];
                result = nn.FeedForward(inputs[i]);
                Console.WriteLine($"expected: {expected} result: {result}");
            }

            Console.ReadLine();
        }
    }
    /*for(int i = 0; i < nn.neuronLayer.Count; i++)
            {
                for (int j = 0; j < nn.neuronLayer[i].Count; j++)
                {
                    for (int k = 0; k < nn.neuronLayer[i][j].WeightsCount; k++)
                    {
                        Console.WriteLine(nn.neuronLayer[i][j].Weights[k]);
                    }
                    Console.WriteLine();
                }
            }*/
}
