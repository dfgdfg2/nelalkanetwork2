using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    internal class Testing
    {
        static void ShowStatisticOfNN(double[][] inputs, int[] outputs, NeuralNetwork nn)
        {
            int length = inputs.GetLength(0);

            double result = 0;
            for (int i = 0; i < length; i++)
            {
                int expected = outputs[i];
                result = nn.FeedForward(inputs[i]);
                Console.WriteLine($"expected: {expected} result: {result}");
            }   
        }
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
            Topology top = new Topology(4, 1, 0.1, new List<int> { 2 }, 10000);
            NeuralNetwork nn = new NeuralNetwork(top);
            List<Regularizations> regs = new List<Regularizations> { Regularizations.Dropout };
            double result = nn.Learn(inputs, outputs, top.Epoch, new GradientDescent(nn, new Adam(top, regs)));

            Console.WriteLine($"avg: {result}");

            ShowStatisticOfNN(inputs, outputs, nn);

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
