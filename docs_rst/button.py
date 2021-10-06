from __future__ import absolute_import
from docutils import nodes
import jinja2
from docutils.parsers.rst.directives import unchanged
from docutils.parsers.rst import Directive

BUTTON_TEMPLATE = jinja2.Template(u"""
<div style="text-align: center;">
<style>

/* CSS */
.button {
  background-color: #2088be;
  border-radius: .5rem;
  box-sizing: border-box;
  color: #FFFFFF;
  display: flex;
  font-size: 20px;
  justify-content: center;
  padding: 1rem 1.75rem;
  text-decoration: none;
  width: 50%;
  border: 0;
  margin-left: auto;
  margin-right: auto;
  cursor: pointer;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
}

.button:hover {
  background-image: linear-gradient(-180deg, #1D95C9 0%, #2088be 100%);
}

@media (min-width: 768px) {
  .button {
    padding: 1rem 2rem;
  }
}
</style>
    <a href="{{ link }}">
        <button class="button">{{ text }}</button>
    </a>
</div>
""")


# placeholder node for document graph
class button_node(nodes.General, nodes.Element):
    pass


class ButtonDirective(Directive):
    required_arguments = 0

    option_spec = {
        'text': unchanged,
        'link': unchanged,
    }

    # this will execute when your directive is encountered
    # it will insert a button_node into the document that will
    # get visisted during the build phase
    def run(self):
        env = self.state.document.settings.env
        app = env.app

        node = button_node()
        node['text'] = self.options['text']
        node['link'] = self.options['link']
        return [node]


# build phase visitor emits HTML to append to output
def html_visit_button_node(self, node):
    html = BUTTON_TEMPLATE.render(text=node['text'], link=node['link'])
    self.body.append(html)
    raise nodes.SkipNode


# if you want to be pedantic, define text, latex, manpage visitors too..

def setup(app):
    app.add_node(button_node,
                 html=(html_visit_button_node, None))
    app.add_directive('button', ButtonDirective)
