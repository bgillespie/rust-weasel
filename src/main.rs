//
// Rust Genetic Algorithm example.
//

extern crate rand;
extern crate num;

use rand::prelude::*;

use std::fmt;
use std::cmp;

const ALPHABET: &str = " AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz";
const TARGET: &str = "Methinks It Is Like A Weasel";
const MUTATE_CHANCE: f64 = 0.01f64;
const POP_SIZE: usize = 10000;


/// The worst possible score against an Individual of the given size.
fn worst_case(alphabet_len: usize, target_len: usize) -> f64 {
    let worst = ((alphabet_len + 1) / 2) as f64;
    pythagoras_distance(std::iter::repeat(&worst).take(target_len))
}


/// Find the index of the value in the given slice of sorted values that is
/// _just_ greater than the search value; the value at index - 1 will be less
/// that the search value. If the search_val is greater than the largest value
/// in the scores, the result will be scores.len(), i.e. one more than the 
/// rightmost index.
fn index_of_container<T>(scores: &[T], search_val: T) -> usize 
        where T: PartialOrd {
    let mut lower = 0;
    let mut upper = scores.len();
    let mut mid;
    while lower < upper {
        mid = (lower + upper) >> 1;
        if scores[mid] < search_val {
            lower = mid + 1;
        }
        else {
            upper = mid;
        }
    }
    lower
}

#[test]
fn test_index_of_container() {
    let cont = [5, 6, 8, 10];
    assert_eq!(0, index_of_container(&cont, 0));
    assert_eq!(0, index_of_container(&cont, 5));
    assert_eq!(2, index_of_container(&cont, 7));
    assert_eq!(3, index_of_container(&cont, 10));
    assert_eq!(4, index_of_container(&cont, 11));

    let cont = [5, 6, 8, 10, 12];
    assert_eq!(0, index_of_container(&cont, 0));
    assert_eq!(0, index_of_container(&cont, 5));
    assert_eq!(2, index_of_container(&cont, 7));
    assert_eq!(3, index_of_container(&cont, 10));
    assert_eq!(5, index_of_container(&cont, 13));
}


/// Given a vector from the origin (as an iterator over numbers), find the
/// scalar distance from the origin using Pythagoras.
fn pythagoras_distance<'a, T: 'a, I>(v: I) -> f64
        where T: num::ToPrimitive,
              T: Copy,
              I: Iterator<Item=&'a T> {
    v.map(|i| i.to_f64().unwrap())
        .fold(0.0f64, |acc, i| acc + i * i)
        .sqrt()
}

#[test]
fn test_pythagoras_distance() {
    let mut v: [f64;2] = [3.0, 4.0];
    assert_eq!(5, pythagoras_distance(v.iter()) as usize);
}


/// Struct representing an Individual.
struct Individual {
    genome: Box<[u8]>,
}

impl Individual {

    /// Create a new Individual.
    fn new(size: usize) -> Self {
        let mut v: Vec<u8> = Vec::with_capacity(size);
        for _ in 0..size {
            v.push(0);
        }
        Individual {
            genome: v.into_boxed_slice(),
        }
    }

    /// Length of the Individual's genome.
    fn len(&self) -> usize {
        self.genome.len()
    }

    /// Randomize the genome of this Individual.
    fn randomize(&mut self, max: usize) {
        let mut rng = thread_rng();
        for index in 0..self.genome.len() {
            self.genome[index] = rng.gen_range(0, max) as u8;
        }
    }

    /// Roll one of the letters of the genome by a given amount.
    fn roll(&mut self, index:usize, amount:isize, max: usize) {
        let max = max as isize;
        let mut p = self.genome[index] as isize;
        p += amount;
        p += (max / amount).abs() * max;
        p %= max;
        self.genome[index] = p as u8;
    }

    /// Crosses over two Individuals to produce an offspring.
    /// Assumes both Individuals are of the same size...
    fn crossover(&self, other: &Individual, index: usize) -> Individual {
        let mut v: Vec<u8> = Vec::with_capacity(self.len());
        v.extend(&self.genome[0..index]);
        v.extend(&other.genome[index..]);
        Individual {
            genome: v.into_boxed_slice()
        }
    }

    /// Distance of each part of this Individual's genome from the
    /// corresponding part of the other Individual's genome.
    fn distances(&self, other: &Individual, max: usize) -> Box<[usize]> {
        self.genome
            .iter()
            .zip(other.genome.iter())
            .map(|(&m, &o)| {
                // Find the shortest number of letters between the
                // actual letter and the desired letter.
                let d = ((m as isize) - (o as isize)).abs() as usize;
                cmp::min(d, max - d)
            })
            .collect::<Vec<usize>>().into_boxed_slice()
    }

    /// Pick a random element of the genome and mutate it.
    fn mutate(&mut self, mutate_chance: f64, max: usize) {
        let mut rng = thread_rng();
        while rand::random::<f64>() < mutate_chance {
            let index = rng.gen_range(0, self.genome.len()) as usize;
            // self.roll(index, if rand::random() {-1} else {1}, max);
            self.genome[index] = rng.gen_range(0, max) as u8;
        }
    }

    /// Mate this Individual with another Individual to produce a new
    /// Individual.
    fn mate(&self, other: &Individual, mutate_chance: f64, max: usize) -> Individual {
        let mut g = self.crossover(
            other,
            rand::thread_rng().gen_range(0, self.genome.len()) as usize
        );
        g.mutate(mutate_chance, max);
        g
    }
}

#[cfg(test)]
mod test_individual {
    use super::{Individual, ALPHABET};

    #[test]
    fn test_roll() {
        let mut individual = Individual::from(" ");
        for _ in 0..ALPHABET.len() + 2 {
            individual.roll(0, -1, ALPHABET.len());
            assert!(individual.genome[0] < ALPHABET.len() as u8);
        }

        let mut individual = Individual::from(" ");
        for _ in 0..ALPHABET.len() {
            individual.roll(0, -1, ALPHABET.len());
            assert!(individual.genome[0] < ALPHABET.len() as u8);
        }
    }

    #[test]
    fn test_crossover() {
        let i1 = Individual::from("Hello David");
        let i2 = Individual::from("Byeee World");
        assert!(format!("{}", i1.crossover(&i2, 5)) == "Hello World");
        assert!(format!("{}", i2.crossover(&i1, 5)) == "Byeee David");
    }

    #[test]
    fn test_distances() {
        let i1 = Individual::from(" z zaA");
        let i2 = Individual::from(" zz Aa");
        let expected = [0, 0, 1, 1, 1, 1, ];
        let actual = i1.distances(&i2, ALPHABET.len());
        // println!("{}", r);
        assert!(actual.iter().zip(expected.iter()).all(|(&a, &b)| a == b));
    }
}

/// So we can print an Individual...
impl fmt::Display for Individual {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let a = ALPHABET.as_bytes();
        let s = self.genome
            .iter()
            .map(|&c| a[c as usize] as char)
            .collect::<String>();
        write!(f, "{}", s)
    }
}

impl From<&str> for Individual {
    fn from(source: &str) -> Self {
        let mut g = Individual::new(source.len());
        g.genome =
            source
                .chars()
                .map(|c| ALPHABET.find(c).unwrap_or_else(|| panic!("{} not in {}", c, ALPHABET)) as u8)
                .collect::<Vec<u8>>()
                .into_boxed_slice();
        g
    }
}

struct World<'a> {
    alphabet     : &'a str,
    target       : &'a Individual,
    worst_case   : f64,
    population   : Box<[Individual]>,
    mutate_chance: f64,
}

impl<'a> World<'a> {

    /// Create a new World
    fn new(size: usize, alphabet: &'a str, target: &'a Individual, mutate_chance: f64) -> World<'a> {

        // this is the most "wrong" score an Individual could possibly get from _any_ target.
        let worst_case = worst_case(alphabet.len(), target.len());

        // this is the initial population
        let mut p: Vec<Individual> = Vec::with_capacity(size);
        for _ in 0..size {
            let mut i = Individual::new(target.len());
            // i.randomize(target.len());
            p.push(i);
        }
        World {
            alphabet,
            target,
            worst_case,
            population: p.into_boxed_slice(),
            mutate_chance,
        }
    }

    /// The size of the population
    fn len(&self) -> usize {
        self.population.len()
    }

    /// For every Individual in the population, generate a corresponding score
    /// in the region 0.0 -> 1.0
    fn score_line(&self) -> Box<[f64]> {
        let len = self.alphabet.len();
        self.population
            .iter()
            .scan(0.0f64, |acc, i| {
                let score = 
                    (1.0 - 
                     pythagoras_distance(i.distances(&self.target, len).iter()) / self.worst_case
                    ).powf(3.0);
                *acc += score;
                if score > 1.0 {
                    panic!("score>1.0");
                }
                Some(*acc)
            })
            .collect::<Vec<f64>>()
            .into_boxed_slice()
    }

    /// Create the next generation.
    fn evolve(&mut self) {

        // Score each individual against a fitness function.
        let scored = self.score_line();

        // Show the best and the worst
        let mut min_i = 0;
        let mut max_i = 0;
        let mut min_s = scored[0];
        let mut max_s = scored[0];
        let mut score:f64;
        for i in 1..self.len() {
            score = scored[i] - scored[i - 1];
            if score < min_s {
                min_s = score;
                min_i = i;
            }
            if score > max_s {
                max_s = score;
                max_i = i;
            }
        }
        println!(
            "Best = {:.*} : {} ; Worst = {:.*} : {}",
            2, max_s, self.population[max_i],
            2, min_s, self.population[min_i]
        );

        // Breed the next population
        let max_score:f64 = scored[scored.len() - 1];
        let mut p: Vec<Individual> = Vec::with_capacity(self.len());
        for _ in 0..self.len() {
            let s1 = rand::random::<f64>() * max_score;
            let s2 = rand::random::<f64>() * max_score;
            let i1 = index_of_container(&scored, s1);
            let i2 = index_of_container(&scored, s2);
            let i = self.population[i1]
                .mate(
                    &self.population[i2],
                    self.mutate_chance,
                    self.alphabet.len()
                );
            p.push(i);
        }
        self.population = p.into_boxed_slice();
    }
}


fn main() {
    // This is the ideal individual for the environment.
    let target = Individual::from(TARGET);
    let mut world = World::new(POP_SIZE, ALPHABET, &target, MUTATE_CHANCE);
    println!("Running");
    for generation in 0..100000000 {
        // Talk about my generation
        print!("{} :", generation);
        world.evolve();
    }
}
