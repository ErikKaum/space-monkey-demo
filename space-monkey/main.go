package main

import (
	"flag"
	"fmt"
	"monkey/evaluator"
	"monkey/lexer"
	"monkey/object"
	"monkey/parser"
	"monkey/repl"
	"os"
)

func main() {
	// Define a command-line flag for code execution
	codeFlag := flag.String("c", "", "Code to execute")
	flag.Parse()

	if *codeFlag != "" {
		// Execute the provided code
		env := object.NewEnvironment()
		l := lexer.New(*codeFlag)
		p := parser.New(l)

		program := p.ParseProgram()
		if len(p.Errors()) != 0 {
			printParserErrors(os.Stdout, p.Errors())
			os.Exit(1)
		}

		evaluated := evaluator.Eval(program, env)
		if evaluated != nil {
			fmt.Println(evaluated.Inspect())
		}
	} else {
		// Start the REPL if no code is provided
		fmt.Println("Welcome Wizard(s) -- Space Monkey 🚀 🐒 v0.0.1")
		repl.Start(os.Stdin, os.Stdout)
	}
}

func printParserErrors(out *os.File, errors []string) {
	fmt.Fprintln(out, "Parser errors:")
	for _, msg := range errors {
		fmt.Fprintln(out, "\t"+msg)
	}
}
