package repl

import (
	"fmt"
	"io"
	"monkey/evaluator"
	"monkey/lexer"
	"monkey/object"
	"monkey/parser"
	"monkey/token"
	"strings"

	"github.com/chzyer/readline"
)

const PROMPT = ">> "

var simpleCommands = []string{
	"let arr = [1,2,3,4]",
	"if (false) { 5 }",
	"if (true) { 5 }",
	"let double = fn(x, add) { add(x,x) }",
	"let add = fn(x,y) { x + y };",
	"let x = 5;",
}

func Start(in io.Reader, out io.Writer) {
	env := object.NewEnvironment()

	rl, err := readline.NewEx(&readline.Config{
		Prompt:          PROMPT,
		InterruptPrompt: "^C",
		EOFPrompt:       "exit",
	})
	if err != nil {
		panic(err)
	}
	defer rl.Close()

	// Add simple commands to readline's history
	for _, cmd := range simpleCommands {
		rl.SaveHistory(cmd)
	}

	commandIndex := -1

	for {
		line, err := rl.Readline()
		if err != nil {
			if err == readline.ErrInterrupt {
				break
			}
			continue
		}

		switch {
		case line == "clear":
			fmt.Print("\033[H\033[2J")
			fmt.Fprint(out, "\033[H\033[2J")
			continue
		case line == "quit":
			return
		case strings.HasPrefix(line, ":lex"):
			input := strings.TrimPrefix(line, ":lex ")
			l := lexer.New(input)
			for tok := l.NextToken(); tok.Type != token.EOF; tok = l.NextToken() {
				fmt.Fprintf(out, "%+v\n", tok)
			}
			continue
		case line == "":
			continue
		case line == string(readline.CharLineStart): // Up arrow
			commandIndex = (commandIndex - 1 + len(simpleCommands)) % len(simpleCommands)
			rl.SetPrompt(PROMPT + simpleCommands[commandIndex])
			rl.Refresh()
			continue
		case line == string(readline.CharLineEnd): // Down arrow
			commandIndex = (commandIndex + 1) % len(simpleCommands)
			rl.SetPrompt(PROMPT + simpleCommands[commandIndex])
			rl.Refresh()
			continue
		}

		// Reset commandIndex and prompt when a command is executed
		commandIndex = -1
		rl.SetPrompt(PROMPT)

		l := lexer.New(line)
		p := parser.New(l)

		program := p.ParseProgram()
		if len(p.Errors()) != 0 {
			printParserErrors(out, p.Errors())
			continue
		}

		evaluated := evaluator.Eval(program, env)
		if evaluated != nil {
			io.WriteString(out, evaluated.Inspect())
			io.WriteString(out, "\n")
		}
	}
}

func printParserErrors(out io.Writer, errors []string) {
	io.WriteString(out, " Errors:\n")
	for _, msg := range errors {
		io.WriteString(out, "\t"+msg+"\n")
	}
}
