package llm

import "sync"

type pool[K comparable] struct {
	devices map[K]bool
	mutex   sync.Mutex
	cond    *sync.Cond
}

func newPool[K comparable](devices ...K) *pool[K] {
	p := &pool[K]{
		devices: make(map[K]bool),
	}

	for _, d := range devices {
		p.devices[d] = true
	}

	p.cond = sync.NewCond(&p.mutex)

	return p
}

func (p *pool[K]) Len() int {
	return len(p.devices)
}

func (p *pool[K]) Acquire() K {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	for {
		for k, v := range p.devices {
			if v {
				p.devices[k] = false

				return k
			}
		}

		p.cond.Wait()
	}
}

func (p *pool[K]) Release(device K) {
	p.mutex.Lock()

	p.devices[device] = true

	p.cond.Signal()
	p.mutex.Unlock()
}
